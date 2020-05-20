# Reconstruct mesh

Use `extract_mesh.ipynb` (the notebook, **not** the py script) to extract **colored** mesh. The guideline for choosing good parameters is commented in the notebook.
Here, I'll give detailed explanation of how it works. There is also a [video](https://youtu.be/t06qu-gXrxA) that explains the same thing.

## Step 1. Predict occupancy

As the [original repo](https://github.com/bmild/nerf/blob/master/extract_mesh.ipynb), we need to first infer which locations are occupied by the object. This is done by first create a grid volume in the form of a cuboid covering the whole object, then use the nerf model to predict whether a cell is occupied or not. This is the main reason why mesh construction is only available for 360 inward-facing scenes as forward facing scenes would require a **huge** volume to cover the whole space! It is computationally impossible to predict the occupancy for all cells.

## Step 2. Perform marching cube algorithm

After we know which cells are occupied, we can use [marching cube algorithm](https://en.wikipedia.org/wiki/Marching_cubes) to extract mesh. This mesh will only contain vertices and faces, if you don't require color, you can stop here and export the mesh. Until here, the code is the same as the original repo.

## Step 3. Remove noise

The mesh might contain some noise, which could be due to wrongly predicted occupancy in step 1, or you might consider the floor as noise. To remove these noises, we use a simple method: only keep the largest cluster. We cluster the triangles into groups (two triangles are in the same group if they are connected), and only keep the biggest one. After removing the noise, we then compute the color for each vertex.

## Step 4. Compute color for each vertex

We adopt the concept of assigning colors to vertices instead of faces (they are actually somehow equivalent, as you can think of the color of vertices as the average color of neighboring faces and vice versa). To compute the color of a vertex, we leverage the **training images**: we project this vertex onto the training images to get its rgb values, then average these values as its final color. Notice that the projected pixel coordinates are floating numbers, and we use *bilinear interpolation* as its rgb value.

This process might seem correct at first sight, however, this is what we'll get:

<img src="https://user-images.githubusercontent.com/11364490/80859055-c0748200-8c98-11ea-9aee-f6cfbd0111f2.png" width=200>

by projecting the vertices onto this input image:

<img src="https://user-images.githubusercontent.com/11364490/80859105-06314a80-8c99-11ea-87e5-c4aeb3831486.png" width=200>

You'll notice the face appears on the mantle. Why is that? It is because of **occlusion**.

From the input image view, that spurious part of the mantle is actually occluded (blocked) by the face, so in reality we **shouldn't** assign color to it, but the above process assigns it the same color as the face because those vertices are projected onto the face (in pixel coordinate) as well!

So the problem becomes: How do we correctly infer occlusion information, to know which vertices shouldn't be assigned colors? I tried two methods, where the first turns out to not work well:

1.  Use depth information

    The first intuitive way is to leverage vertices' depths (which is obtained when projecting vertices onto image plane): if two (or more) vertices are projected onto the **same** pixel coordinates, then only the nearest vertex will be assigned color, the rest remains untouched. However, this method won't work since no any two pixels will be projected onto the exact same location! As we mentioned earlier, the pixel coordinates are floating numbers, so it is impossible for they to be exactly the same. If we round the numbers to integers (which I tried as well), then this method works, but with still a lot of misclassified (occluded/non occluded) vertices in my experiments.
    
2.  Leverage NeRF model

    What I find a intelligent way to infer occlusion is by using NeRF model. Recall that nerf model can estimate the opacity (or density) along a ray path (the following figure c):    
    ![nerf](https://github.com/bmild/nerf/blob/master/imgs/pipeline.jpg)
    We can leverage that information to tell if a vertex is occluded or not. More concretely, we form rays originating from the camera origin, destinating (ending) at the vertices, and compute the total opacity along these rays. If a vertex is not occluded, the opacity will be small; otherwise, the value will be large, meaning that something lies between the vertex and the camera.
    
    After applying this method, this is what we get (by projecting the vertices onto the input view as above):
    <img src="https://user-images.githubusercontent.com/11364490/80859510-945b0000-8c9c-11ea-888a-a01ad1c3433d.png" width=200>
    
    The spurious face on the mantle disappears, and the colored pixels are almost exactly the ones we can observe from the image. By default we set the vertices to be all black, so a black vertex means it's occluded in this view, but will be assigned color when we change to other views.


# Finally...

This is the final result:

<img src="https://user-images.githubusercontent.com/11364490/80813184-83f74680-8c04-11ea-8606-40580f753355.png" height="252">

We can then export this `.ply` file to any other format, and embed in programs like I did in Unity:
(I use [this plugin](https://github.com/kwea123/Pcx) to import `.ply` file to unity, and made some modifications so that it can also read mesh triangles, not only the points)

![image](https://user-images.githubusercontent.com/11364490/80859833-9e7dfe00-8c9e-11ea-9fa1-ec48237e3873.png)

The meshes can be attached a meshcollider so that they can interact with other objects. You can see [this video](https://youtu.be/I2M0xhnrBos) for a demo.

## Further reading
The author suggested [another way](https://github.com/bmild/nerf/issues/44#issuecomment-622961303) to extract color, in my experiments it doesn't turn out to be good, but the idea is reasonable and interesting. You can also test this by setting `--use_vertex_normal`.
