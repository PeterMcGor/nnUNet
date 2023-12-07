from typing import Union, Tuple, List
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.blob_loss import BlobLoss
from nnunetv2.training.data_augmentation.custom_transforms.connected_components import (
    ConnectedComponents,
    KEYS,
)
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.blob_loss import BlobLoss


def add_connected_components_transf(
    transforms: Compose,
    transform_keys: List[str] = ["data", "target", KEYS.COMPONENTS],
    convert_type: str = "float",
    # as_extra_target_channel:bool=False,
    **kwargs,
):
    """
    Adds the ConnectedComponents transformation to the given Compose object.

    Args:
        transforms: The Compose object to which the transformations will be added.
        as_extra_target_channel: Adds the conected components (Instace segmentation) as an extra dimension at the 'target' [BS(Semantic Segmentation), BS(Instance Segmentation), C, D,W,H]
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
        add_connected_components_transf(
            tr_transforms,
            transform_keys=["data", "target"],
            as_extra_target_channel=True,
        )
        return tr_transforms

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        # Check the valid comments in get_training_transforms
        add_connected_components_transf(
            val_transforms,
            transform_keys=["data", "target"],
            as_extra_target_channel=True,
        )
        return val_transforms

    def train_step(self, batch: dict) -> dict:
        # I could use a call to the method from the mother class and recover the loss but is was convinient for debugging reasons this one
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            # del data
            #output[0].retain_grad()
            #output[1].retain_grad()
            l = self.loss(output, target)
            #print("***Grads AfterLoss LOSS blob***",torch.mean(output[0].grad),output[0].grad.shape,
            #    torch.mean(output[1].grad),output[1].grad.shape,l.shape,torch.mean(torch.tensor(self.loss.loss.blob_loss_mean)),target[0].shape)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            #print("Grads AfterLoss LOSS", torch.mean(output[0].grad))
            self.grad_scaler.unscale_(self.optimizer)
            #print("Grads AfterLoss LOSS2", torch.mean(output[0].grad))
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": l.detach().cpu().numpy()
            + torch.mean(torch.tensor(self.loss.loss.blob_loss_mean)).cpu().numpy()
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0][0]  # Just the semantic part

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy()  + torch.mean(torch.tensor(self.loss.loss.blob_loss_mean)).cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def _build_loss(self):
        assert (
            not self.label_manager.has_regions
        ), "regions not supported by this trainer"
        dim = len(self.configuration_manager.patch_size)
        loss = super()._build_loss()
        # print("DEEP SUPERVISION SCALES",self._get_deep_supervision_scales(), self.configuration_manager.patch_size)
        if isinstance(loss, DeepSupervisionWrapper):
            #print("loss.weight_factors", loss.weight_factors)
            weights = 1 if loss.weight_factors is None else loss.weight_factors.copy()
            loss.loss = BlobLoss(
                global_loss_criterium=loss.loss,
                blob_loss_criterium=loss.loss,
                trainer=self,
                scale_weights=weights,
            )
        else:
            loss = BlobLoss(
                global_loss_criterium=loss, blob_loss_criterium=loss, trainer=self
            )
        return loss
