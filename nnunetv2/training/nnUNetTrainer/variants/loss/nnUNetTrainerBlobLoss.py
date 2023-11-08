from typing import Union, Tuple, List
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.blob_loss import BlobLoss
from nnunetv2.training.data_augmentation.custom_transforms.connected_components import ConnectedComponents, KEYS

import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.blob_loss import BlobLoss

def add_connected_components_transf(transforms: Compose, 
                                    transform_keys: List[str] = ['data', 'target', KEYS.COMPONENTS], convert_type:str = 'float',
                                    **kwargs):
    
    """
    Adds the ConnectedComponents transformation to the given Compose object.

    Args:
        transforms: The Compose object to which the transformations will be added.
        **kwargs: Keyword arguments specifying the parameters for the ConnectedComponents transformation.
            Supported keyword arguments:
            - seg_key (str): The key for accessing the segmentation mask in the data_dict.
            - output_key (str): The key under which the labeled components will be stored in the data_dict.
            - n_components_key (str): The key under which the number of components per label will be stored in the data_dict.
            - transform_keys (List[str]): The keys for accessing the data, target, and labeled components in the data_dict.
            - convert_type (str): The type to which the arrays will be converted (e.g., 'float', 'double').
            - connectivity (int): The connectivity value used for the connected components labeling. Default = None
    """
    # Get rid of the last tranformation since should be the one to convert the arrays to tensors
    transforms.transforms.pop()
    # Add the Transformations to enable loss per component (blob loss)
    transforms.transforms.append(ConnectedComponents(**kwargs))
    # Finally add the tranformation array --> Tensor
    transforms.transforms.append(NumpyToTensor(transform_keys, convert_type))

class nnUNetTrainerBlobLoss(nnUNetTrainer):
    @staticmethod
    def get_training_transforms(*args, **kwargs) -> AbstractTransform:
        tr_transforms = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        add_connected_components_transf(tr_transforms)
        return tr_transforms
        
    
    @staticmethod
    def get_validation_transforms2(deep_supervision_scales: Union[List, Tuple],
                                  is_cascaded: bool = False,
                                  foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                  regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                  ignore_label: int = None) -> AbstractTransform:
        
        val_transforms = nnUNetTrainer.get_validation_transforms(deep_supervision_scales,
                                          is_cascaded=is_cascaded,
                                          foreground_labels=foreground_labels,
                                          regions=regions,
                                          ignore_label=ignore_label)
        # Check the valid comments in get_training_transforms
        add_connected_components_transf(val_transforms, output_key='target', transform_keys= ['data', 'target'])
        return val_transforms
    
    def train_step(self, batch: dict) -> dict:
        # nnUNetTrainer call the loss by retriving the 'target' data in each yielded batch and nothing else.  [Line 854](#nnUNetTrainer-train_step-L854) 
        # in this method performs a crucial calculation. This way there isno need to make extra modifications
        if isinstance(batch['target'], list): 
            #conc_targ = [torch.cat((t, batch[KEYS.COMPONENTS][i]), dim=1) for i,t in enumerate(batch['target'])]
            conc_targ = [torch.stack((t, batch[KEYS.COMPONENTS][i]), dim=0) for i,t in enumerate(batch['target'])]
        else:
            #conc_targ = torch.cat(batch['target'], batch[KEYS.COMPONENTS], dim=1)
            conc_targ = torch.stack([batch['target'], batch[KEYS.COMPONENTS]], dim=0)
        return super().train_step({'data':batch['data'], 'target':conc_targ})
        
        
    def _build_loss(self):
        assert not self.label_manager.has_regions, 'regions not supported by this trainer'
        dim = len(self.configuration_manager.patch_size)
        loss = super()._build_loss()
        if isinstance(loss, DeepSupervisionWrapper):
            loss.loss = BlobLoss(global_loss_criterium=loss.loss, blob_loss_criterium=loss.loss, expected_dim=dim)
        else: 
            loss = BlobLoss(global_loss_criterium=loss, blob_loss_criterium=loss, expected_dim=dim)
        return loss
        


