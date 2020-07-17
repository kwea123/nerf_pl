import torch
from torchsearchsorted import searchsorted

__all__ = ['render_rays']

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""

def normalize(tensor):
    """
    tensor: (B, 3)
    """
    norm = torch.norm(tensor, dim=1, keepdim=True)
    return tensor / (norm+1e-6)


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz_, dir_, dir_embedded, z_vals, test_time=False):
        """
        Helper function that performs model inference.

        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if not (typ=='coarse' and test_time): # infer rgb and sigma and others
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                           # (N_rays*N_samples_, embed_dir_channels)
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i+chunk]], 1)
                out_chunks += [model(xyzdir_embedded, sigma_only=False)]

            out = torch.cat(out_chunks, 0)
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)
        else:
            for i in range(0, B, chunk):
                xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                out_chunks += [model(xyz_embedded, sigma_only=True)]

            out = torch.cat(out_chunks, 0)
            sigmas = out.view(N_rays, N_samples_)

        if typ == 'fine':
            if not test_time:  # regularize normals in fine model
                subsample_idx = torch.randint(B, (N_rays,), device=xyz_.device)
                xyz_subsampled = xyz_[subsample_idx] # (N_rays, 3)
                neighbor_dist = 1e-4
                xyz_subsampled_neighbors = xyz_subsampled + \
                                torch.rand_like(xyz_subsampled)*neighbor_dist - neighbor_dist/2

                normals_ndc_chunks = []
                normals_ndc_neighbors_chunks = []
                for i in range(0, N_rays, chunk):
                    # Compute normals using finite difference
                    normals_ndc_chunk = []
                    for j in range(3):
                        eps = torch.zeros(3, device=xyz_.device)
                        eps[j] = 1e-6
                        xyz_embedded_m = embedding_xyz(xyz_subsampled[i:i+chunk]-eps)
                        xyz_embedded_p = embedding_xyz(xyz_subsampled[i:i+chunk]+eps)
                        df_dxj = model(xyz_embedded_m, sigma_only=True) - \
                                 model(xyz_embedded_p, sigma_only=True)
                        normals_ndc_chunk += [df_dxj] # (chunk, 1)
                    normals_ndc_chunks += [torch.cat(normals_ndc_chunk, 1)] # (chunk, 3)

                    normals_ndc_neighbors_chunk = []
                    for j in range(3):
                        eps = torch.zeros(3, device=xyz_.device)
                        eps[j] = 1e-6
                        xyz_embedded_m = embedding_xyz(xyz_subsampled_neighbors[i:i+chunk]-eps)
                        xyz_embedded_p = embedding_xyz(xyz_subsampled_neighbors[i:i+chunk]+eps)
                        df_dxj = model(xyz_embedded_m, sigma_only=True) - \
                                 model(xyz_embedded_p, sigma_only=True)
                        normals_ndc_neighbors_chunk += [df_dxj] # (chunk, 1)
                    normals_ndc_neighbors_chunks += [torch.cat(normals_ndc_neighbors_chunk, 1)] # (chunk, 3)

                normals_ndc = torch.cat(normals_ndc_chunks, 0) # (N_rays, 3)
                normals_ndc = normalize(normals_ndc)
                normals_ndc_neighbors = torch.cat(normals_ndc_neighbors_chunks, 0) # (N_rays, 3)
                normals_ndc_neighbors = normalize(normals_ndc_neighbors)

            else:
                normals_ndc_chunks = []
                for i in range(0, B, chunk):
                    # Compute normals using finite difference
                    normals_ndc_chunk = []
                    for j in range(3):
                        eps = torch.zeros(3, device=xyz_.device)
                        eps[j] = 1e-6
                        xyz_embedded_m = embedding_xyz(xyz_[i:i+chunk]-eps)
                        xyz_embedded_p = embedding_xyz(xyz_[i:i+chunk]+eps)
                        df_dxj = model(xyz_embedded_m, sigma_only=True) - \
                                 model(xyz_embedded_p, sigma_only=True)
                        normals_ndc_chunk += [df_dxj] # (chunk, 1)
                    normals_ndc_chunks += [torch.cat(normals_ndc_chunk, 1)] # (chunk, 3)
                normals_ndc = torch.cat(normals_ndc_chunks, 0) # (N_rays*N_samples_, 3)
                normals_ndc = normalize(normals_ndc)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                     # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f'weights_{typ}'] = weights
        if test_time and typ == 'coarse':
            return

        # compute final weighted outputs
        rgb_map = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_map = torch.sum(weights*z_vals, -1) # (N_rays)

        if white_back:
            rgb_map = rgb_map + 1-weights_sum.unsqueeze(-1)

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map
        if typ == 'fine':
            if not test_time:
                results['normals_ndc_fine'] = normals_ndc
                results['normals_ndc_neighbors_fine'] = normals_ndc_neighbors
            else:
                normals_ndc = normals_ndc.view(N_rays, N_samples_, 3)
                normal_map = torch.sum(weights.unsqueeze(-1)*normals_ndc, -2) # (N_rays, 3)
                results['normal_map_fine'] = normalize(normal_map)

        return

    # Extract models from lists
    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Embed direction
    dir_embedded = embedding_dir(rays_d) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    results = {}
    inference(results, model_coarse, 'coarse', xyz_coarse_sampled, rays_d,
              dir_embedded, z_vals, test_time)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)
                    # values are interleaved actually, so maybe can do better than sort?

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
                           # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        inference(results, model_fine, 'fine', xyz_fine_sampled, rays_d,
                  dir_embedded, z_vals, test_time)

    return results