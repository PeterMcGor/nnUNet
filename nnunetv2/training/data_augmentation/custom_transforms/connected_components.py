from typing import List, Tuple, Union
from skimage.measure import label
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np


# Define a class to hold constant key values
class KEYS:
    COMPONENTS = "compon"
    N_COMPONETS = "n_compon"


class ConnectedComponents(AbstractTransform):
    def __init__(
        self,
        seg_key: str = "target",
        as_extra_target_channel:bool=False,
        output_key: str = KEYS.COMPONENTS,
        n_componnets_key: str = KEYS.N_COMPONETS,
        connectivity: int = None,
    ):
        """
        Args:
            seg_key: The key for accessing the segmentation mask in the data_dict.
            output_key: The key under which the labeled components will be stored in the data_dict.
            n_components_key: The key under which the number of components per label will be stored in the data_dict.
            connectivity: The connectivity value used for the connected components labeling.

        """
        self.output_key = output_key
        self.n_comp = n_componnets_key
        self.seg_key = seg_key
        self.connectivity = connectivity
        self.extra_target_channel = as_extra_target_channel

    def __call__(self, **data_dict):
        # Assumes that the segmentation is going to be a single mask (channel), so [B,C,D,W,H], C=1, index 0
        labeled_image = []
        unique_components = []
        # Deep Supervision(DS) requires to compare the mask at each decoded resolution/level.
        # nnUNet implements DS by including at the preprocessing/DA all the needed downsampled targets (masks)
        # I need to apply connected componets to all downed-resolution targets to apply DS (if wanted) as well
        deep_scales = isinstance(data_dict[self.seg_key], list)
        for i, seg in enumerate(
            data_dict[self.seg_key]
        ):  # DS loop if DS otherwise batch loop
            seg = (
                np.expand_dims(seg, axis=0) if not deep_scales else seg
            )  # just for consistance and code reducing
            conected_comps = np.zeros_like(seg)
            n_comps = []
            # batch loop
            for j, deep_scale in enumerate(seg):
                con, n_con = label(
                    deep_scale[0],
                    return_num=True,
                    connectivity=self.connectivity,  # label does not work in more than 3D
                )
                conected_comps[j] = con
                n_comps.append(n_con)
            labeled_image.append(conected_comps)
            unique_components.append(n_comps)

        labeled_image = (
            labeled_image
            if deep_scales
            else np.concatenate(
                labeled_image, axis=0
            )  # DS need a list with the mask at each resolution, No DS just the normal batch
        )
        if self.extra_target_channel:
            if deep_scales:
                data_dict[self.seg_key] = [np.stack((t, labeled_image[i]), axis=0) for i,t in enumerate(data_dict[self.seg_key])]
            else:
                data_dict[self.seg_key] = np.stack([batch['target'], labeled_image], axis=0)
        else:
            data_dict[self.output_key] = labeled_image
            
        data_dict[self.n_comp] = unique_components
        return data_dict
