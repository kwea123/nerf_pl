# Generating files for Unity rendering

This readme contains guidances for generating files required for Unity rendering in my [Unity project](https://github.com/kwea123/nerf_Unity)

## MeshRender

See [README_mesh](README_mesh.md) for generating mesh.
You then need [this plugin](https://github.com/kwea123/Pcx) to import `.ply` files into Unity.

## MixedReality

Use `eval.py` with `--save_depth --depth_format bytes`to create the whole sequence of moving views. E.g.
```
python eval.py \
   --root_dir $BLENDER \
   --dataset_name blender --scene_name lego \
   --img_wh 400 400 --N_importance 64 --ckpt_path $CKPT_PATH \
   --save_depth --depth_format bytes
```
You will get `*.png` files and corresponding `depth_*` files. Now import the image you want to show and its corresponding depth file into Unity, and replace the files in my Unity project.

## VolumeRender

Use `extract_mesh.ipynb` (not `extract_color_mesh.py`!) to find the tight bounds for the object as for mesh generation (See [this video](https://www.youtube.com/watch?v=t06qu-gXrxA&t=1355)), but this time stop before the cell "Extract colored mesh". Remember to set `N=512` in the cell "Search for tight bounds of the object" and comment out the lines for visualization. Now run the cell "Generate .vol file for volume rendering in Unity", after that, you should obtain a `.vol` file, which you can import to my Unity project and render.

**NOTE:** If you use colab as in the video, copy the cell "Generate .vol file for volume rendering in Unity" into colab notebook and execute it.

If you want to render in your own project, you need the script [LoadVolume.cs](https://github.com/kwea123/nerf_Unity/blob/master/Assets/Editor/LoadVolume.cs) which reads this own-defined `.vol` into a `Texture3D`.
