from collections.abc import Iterable

import torch
from torch import nn, Tensor
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.connected_components import MergeSemanticInstace


class BlobLoss(nn.Module):
    """ """

    def __init__(
        self,
        global_loss_criterium: nn.Module,
        blob_loss_criterium: nn.Module,
        global_weight: float = 1.0,
        blob_weight: float = 2.0,
        trainer: nnUNetTrainer = None,
        scale_weights=1,
    ):
        super(BlobLoss, self).__init__()
        self.global_loss = global_loss_criterium
        self.blob_loss = blob_loss_criterium
        self.global_weight = global_weight
        self.blob_weight = blob_weight
        self.trainer = trainer if trainer is not None else None
        #self.scales_blob_loses = []
        #self.scales_global_losses = []
        
        with torch.no_grad():
            self.scale_weights = scale_weights
            self.blob_loss_mean = []
            self.merger = MergeSemanticInstace()
        # self.expected_dim = expected_dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """ """
        # This is horrible but the framework doesnt give me many choices
        training = (
            input.requires_grad
        )  # len(target.shape) == self.expected_dim + 3 # (extra labelled, BS, C)
        #print("Target Original Shape", target.shape, training)
        #labeled = target[1]  # Instance segmentation GT
        #target = target[0]  # Semantic Segmentation GT
        with torch.no_grad():
            target, labeled = self.merger.unmerge(target.long())
            unique_components = torch.unique(labeled)
        #print("LOSS UNIquE",unique_components, torch.unique(target))

        #if len(target.shape) == len(input.shape):
        #    assert target.shape[1] == 1

        global_loss = self.global_loss(input, target.long())
        #print("global_loss",global_loss)
        #ith torch.no_grad():
        #    unique_components = torch.unique(labeled)

        # print("UNIQUES Train", labeled.shape, input.shape,torch.unique(labeled), torch.unique(target), torch.unique(target.long()), self.trainer.grad_scaler.get_scale(), unique_components)
        input_detached = input.detach()
        input_detached.requires_grad = True
        #input_detached.retain_grad() #TODO do I need this?
        dims = list(range(1, labeled.dim()))

        for component in unique_components[1:]:  # Avoid the bck
            # assumes the first dimension as the batch size
            sample_contain_component = (labeled == component).sum(dim=dims) > 0
            # get just the blob envioroment. Mask whatever correspond to other blobs in the GT. Not the background!
            masked_target, masked_output = self.mask_all_labels(
                labeled[sample_contain_component],
                input_detached[sample_contain_component],
                label_to_keep=component,
            )
            b = 0
            blob_loss_1 = []
            for i, sample in enumerate(
                sample_contain_component
            ):  # per sample in the batch
                if sample:
                    blob_loss = self.blob_loss(
                        masked_output[b].unsqueeze(0), masked_target[b].unsqueeze(0)
                    )
                    deep_supervision_scale = self.get_scale_weight(
                        masked_target.shape[2:]
                    )
                    blob_loss_1.append(
                        blob_loss * self.blob_weight * deep_supervision_scale
                    )
                    b += 1
                else:
                    pass
            blob_loss_1 = (
                torch.mean(torch.stack([sample_loss for sample_loss in blob_loss_1]))
                if len(blob_loss_1) > 0
                else torch.tensor([0.0], device=target.device, requires_grad=True)
            )

            # print("blob_loss mean", blob_loss_1)

            # Here I do the "trick" to free the memory. I can acummulate the gradeients in the inputs.
            # These are finally "zero_grad" at [Line 864](#nnUNetTrainer-train_step-L877)
            # grad scaler is needed incase of small loses which grads could not be represented in 16bits
            if training: 
                if self.trainer.grad_scaler is not None:
                    self.trainer.grad_scaler.scale(blob_loss_1).backward(
                        inputs=input_detached
                    )
                else:
                    blob_loss_1.backward(inputs=input_detached)
    
                input.grad = input.grad + input_detached.grad if input.grad is not None else input_detached.grad
                #print("INPUT GRAD",component, torch.mean(input.grad))
                
            #with torch.no_grad():
            #    self.blob_loss_mean.append(blob_loss_1)  # TODO reporting per instance
                
            #del masked_target, masked_output,blob_loss_1
            #torch.cuda.empty_cache()

        # Just returning this to calculate ther part of the gradient due to the semantic loss because the blob loss  part is apllied inside to avid the memory problemsreturn self.global_weight*self.global_loss(input, target.long())
        #del input_detached, labeled, target
        #torch.cuda.empty_cache()
        return self.global_weight * global_loss

    def get_scale_weight(
        self, patch_size
    ):  # this is all a work arounf to accumulate gradient in DS mode
        with torch.no_grad():
            patch_size = list(patch_size)
            if not isinstance(self.scale_weights, Iterable):
                return self.scale_weights
            for i, scale in enumerate(self.trainer._get_deep_supervision_scales()):
                if np.all(
                    np.array(self.trainer.configuration_manager.patch_size) * scale
                    == patch_size
                ):
                    return self.scale_weights[i]

    @staticmethod
    def mask_all_labels(
        components_target: Tensor,
        apply_mask_to: Tensor,
        label_to_keep: int,
        bck_label: int = 0,
    ):
        # Could be more efficient since the bck is always mostly the same
        with torch.no_grad():
            # WE just mask the rest of the blobs but not the background
            mask = (components_target == bck_label) | (
                components_target == label_to_keep
            )
            mask = torch.where(mask, 1, 0)
            component_mask = ((components_target * mask) > 0).to(
                components_target.dtype
            )
        apply_mask_to = apply_mask_to * mask
        del mask
        torch.cuda.empty_cache()
        return component_mask, apply_mask_to
