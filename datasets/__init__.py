from .blender import BlenderDataset
from .llff import LLFFDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset}