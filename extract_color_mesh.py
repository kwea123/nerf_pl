import torch
import os
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm
import mcubes
import open3d as o3d
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

from models.rendering import *
from models.nerf import *

from utils import load_ckpt

from datasets import dataset_dict

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output ply filename')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of samples to infer the acculmulated opacity')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--N_grid', type=int, default=256,
                        help='size of the grid on 1 side, larger=higher resolution')
    parser.add_argument('--x_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='x range of the object')
    parser.add_argument('--sigma_threshold', type=float, default=20.0,
                        help='threshold to consider a location is occupied')
    parser.add_argument('--occ_threshold', type=float, default=0.2,
                        help='''threshold to consider a vertex is occluded.
                                larger=fewer occluded pixels''')

    #### method using vertex normals ####
    parser.add_argument('--use_vertex_normal', action="store_true",
                        help='use vertex normals to compute color')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of fine samples to infer the acculmulated opacity')
    parser.add_argument('--near_t', type=float, default=1.0,
                        help='the near bound factor to start the ray')

    return parser.parse_args()


@torch.no_grad()
def f(models, embeddings, rays, N_samples, N_importance, chunk, white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        False,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()

    kwargs = {'root_dir': args.root_dir,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':
        kwargs['spheric_poses'] = True
        kwargs['split'] = 'test'
    else:
        kwargs['split'] = 'train'
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    embeddings = [embedding_xyz, embedding_dir]
    nerf_fine = NeRF()
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
    nerf_fine.cuda().eval()

    # define the dense grid for query
    N = args.N_grid
    xmin, xmax = args.x_range
    ymin, ymax = args.y_range
    zmin, zmax = args.z_range
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
           # sigma is independent of direction, so any value here will produce the same result

    # predict sigma (occupancy) for each grid location
    print('Predicting occupancy ...')
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, args.chunk)):
            xyz_embedded = embedding_xyz(xyz_[i:i+args.chunk]) # (N, embed_xyz_channels)
            dir_embedded = embedding_dir(dir_[i:i+args.chunk]) # (N, embed_dir_channels)
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)
            out_chunks += [nerf_fine(xyzdir_embedded)]
        rgbsigma = torch.cat(out_chunks, 0)

    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0).reshape(N, N, N)

    # perform marching cube algorithm to retrieve vertices and triangle mesh
    print('Extracting mesh ...')
    vertices, triangles = mcubes.marching_cubes(sigma, args.sigma_threshold)

    ##### Until mesh extraction here, it is the same as the original repo. ######

    vertices_ = (vertices/N).astype(np.float32)
    ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (ymax-ymin) * vertices_[:, 1] + ymin
    y_ = (xmax-xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax-zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'), 
             PlyElement.describe(face, 'face')]).write(f'{args.scene_name}.ply')

    # remove noise in the mesh by keeping only the biggest cluster
    print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(f"{args.scene_name}.ply")
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.')

    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)

    # perform color prediction
    # Step 0. define constants (image width, height and intrinsics)
    W, H = args.img_wh
    K = np.array([[dataset.focal, 0, W/2],
                  [0, dataset.focal, H/2],
                  [0,             0,   1]]).astype(np.float32)

    # Step 1. transform vertices into world coordinate
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1) # (N, 4)

    if args.use_vertex_normal: ## use normal vector method as suggested by the author.
                               ## see https://github.com/bmild/nerf/issues/44
        mesh.compute_vertex_normals()
        rays_d = torch.FloatTensor(np.asarray(mesh.vertex_normals))
        near = dataset.bounds.min() * torch.ones_like(rays_d[:, :1])
        far = dataset.bounds.max() * torch.ones_like(rays_d[:, :1])
        rays_o = torch.FloatTensor(vertices_) - rays_d * near * args.near_t

        nerf_coarse = NeRF()
        load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
        nerf_coarse.cuda().eval()

        results = f([nerf_coarse, nerf_fine], embeddings,
                    torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                    args.N_samples,
                    args.N_importance,
                    args.chunk,
                    dataset.white_back)

    else: ## use my color average method. see README_mesh.md
        ## buffers to store the final averaged color
        non_occluded_sum = np.zeros((N_vertices, 1))
        v_color_sum = np.zeros((N_vertices, 3))

        # Step 2. project the vertices onto each training image to infer the color
        print('Fusing colors ...')
        for idx in tqdm(range(len(dataset.image_paths))):
            ## read image of this pose
            image = cv2.imread(dataset.image_paths[idx])[:,:,::-1]
            image = cv2.resize(image, tuple(args.img_wh))

            ## read the camera to world relative pose
            P_c2w = np.concatenate([dataset.poses[idx], np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            P_w2c = np.linalg.inv(P_c2w)[:3] # (3, 4)
            ## project vertices from world coordinate to camera coordinate
            vertices_cam = (P_w2c @ vertices_homo.T) # (3, N) in "right up back"
            vertices_cam[1:] *= -1 # (3, N) in "right down forward"
            ## project vertices from camera coordinate to pixel coordinate
            vertices_image = (K @ vertices_cam).T # (N, 3)
            depth = vertices_image[:, -1:]+1e-5 # the depth of the vertices, used as far plane
            vertices_image = vertices_image[:, :2]/depth
            vertices_image = vertices_image.astype(np.float32)
            vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W-1)
            vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H-1)

            ## compute the color on these projected pixel coordinates
            ## using bilinear interpolation.
            ## NOTE: opencv's implementation has a size limit of 32768 pixels per side,
            ## so we split the input into chunks.
            colors = []
            remap_chunk = int(3e4)
            for i in range(0, N_vertices, remap_chunk):
                colors += [cv2.remap(image, 
                                    vertices_image[i:i+remap_chunk, 0],
                                    vertices_image[i:i+remap_chunk, 1],
                                    interpolation=cv2.INTER_LINEAR)[:, 0]]
            colors = np.vstack(colors) # (N_vertices, 3)
            
            ## predict occlusion of each vertex
            ## we leverage the concept of NeRF by constructing rays coming out from the camera
            ## and hitting each vertex; by computing the accumulated opacity along this path,
            ## we can know if the vertex is occluded or not.
            ## for vertices that appear to be occluded from every input view, we make the
            ## assumption that its color is the same as its neighbors that are facing our side.
            ## (think of a surface with one side facing us: we assume the other side has the same color)

            ## ray's origin is camera origin
            rays_o = torch.FloatTensor(dataset.poses[idx][:, -1]).expand(N_vertices, 3)
            ## ray's direction is the vector pointing from camera origin to the vertices
            rays_d = torch.FloatTensor(vertices_) - rays_o # (N_vertices, 3)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            near = dataset.bounds.min() * torch.ones_like(rays_o[:, :1])
            ## the far plane is the depth of the vertices, since what we want is the accumulated
            ## opacity along the path from camera origin to the vertices
            far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
            results = f([nerf_fine], embeddings,
                        torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                        args.N_samples,
                        0,
                        args.chunk,
                        dataset.white_back)
            opacity = results['opacity_coarse'].cpu().numpy()[:, np.newaxis] # (N_vertices, 1)
            opacity = np.nan_to_num(opacity, 1)

            non_occluded = np.ones_like(non_occluded_sum) * 0.1/depth # weight by inverse depth
                                                                    # near=more confident in color
            non_occluded += opacity < args.occ_threshold
            
            v_color_sum += colors * non_occluded
            non_occluded_sum += non_occluded

    # Step 3. combine the output and write to file
    if args.use_vertex_normal:
        v_colors = results['rgb_fine'].cpu().numpy() * 255.0
    else: ## the combined color is the average color among all views
        v_colors = v_color_sum/non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr+v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]
        
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertex_all, 'vertex'), 
             PlyElement.describe(face, 'face')]).write(f'{args.scene_name}.ply')

    print('Done!')
