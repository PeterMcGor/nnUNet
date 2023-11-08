import torch
from torch import nn, Tensor
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
import numpy as np

import SimpleITK as sitk

class BlobLoss(nn.Module):
    """
    """
    def __init__(self, global_loss_criterium: nn.Module, blob_loss_criterium: nn.Module, 
                 global_weight: float = 1.0, blob_weight: float = 2.0, expected_dim = 3):
        super(BlobLoss, self).__init__()
        self.global_loss = global_loss_criterium
        self.blob_loss = blob_loss_criterium
        self.global_weight = global_weight
        self.blob_weight = blob_weight
        self.expected_dim = expected_dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        """
        # This is horrible but the framework doesnt give me many choices
        # TODO a different valaditaion step (change train step to not add the extra labelled)
        training = len(target.shape) == self.expected_dim + 3 # (extra labelled, BS, C)
        #print("LOSS shape", target.shape, self.expected_dim, training)
       
        if training: 
            #labeled = target[:,1].unsqueeze(1)
            #target = target[:,0].unsqueeze(1)
            labeled = target[1]
            target = target[0]
        else: 
            return self.global_loss(input, target.long())
        
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1

        global_loss = self.global_loss(input, target.long())
        
        with torch.no_grad():
            unique_components = torch.unique(labeled)
            
        input_detached = input.detach()
        input_detached.requires_grad = True
        dims = list(range(1,labeled.dim()))
        #imag_id = np.random.randint(10000, 100000)
        for component in unique_components[1:]: # Avoid the bck
            # assumes the first dimension as the batch size
            sample_contain_component = (labeled==component).sum(dim=dims)>0
            # get just the blob envioroment. Mask whatever correspond to other blobs in the GT. Not the background!
            masked_target, masked_output = self.mask_all_labels(labeled[sample_contain_component], input_detached[sample_contain_component], label_to_keep=component)
            b = 0
            blob_loss_1  = []
            for i,sample in enumerate(sample_contain_component):  
                if sample:
                    #sitk.WriteImage( sitk.GetImageFromArray(masked_target[b][0].detach().cpu().numpy()), '/tmp/id_'+str(imag_id)+'_comp_'+str(component.item())+'_b'+str(i)+str(masked_target[b].shape)+'.nii.gz')
                    #sitk.WriteImage( sitk.GetImageFromArray(masked_output[b][0].detach().cpu().numpy().astype(np.float32)), '/tmp/id_'+str(imag_id)+'_netOut_'+str(component.item())+'_b'+str(i)+str(masked_target[b].shape)+'.nii.gz')
                    blob_loss = self.blob_loss(masked_output[b].unsqueeze(0),  masked_target[b].unsqueeze(0)) 
                    blob_loss_1.append(blob_loss)
                    b+=1
                else:
                    pass
            blob_loss_1 = torch.mean(torch.stack([sample_loss for sample_loss in blob_loss_1])) if len(blob_loss_1)>0 else torch.tensor([0.0], device=target.device, requires_grad=True) * self.blob_weight
            
            # Here I do "trick" to free the memory. I can acummulate the gradeients in the inputs. 
            # These are finally "zero_grad" at [Line 864](#nnUNetTrainer-train_step-L864) 
            blob_loss_1.backward(inputs=input_detached)
            input.grad = input.grad +  input_detached.grad if input.grad is not None else input_detached.grad
            #print(component ,"Grad acc.",  input.grad.sum())
            

        return self.global_weight * global_loss 
    
    @staticmethod
    def mask_all_labels(components_target:Tensor, apply_mask_to: Tensor, 
                        label_to_keep: int, bck_label:int = 0):
        # Could be more efficient since the bck is always mostly the same
        with torch.no_grad():
            # WE just mask the rest of the blobs but not the background
            mask = (components_target == bck_label) | (components_target == label_to_keep)
            mask = torch.where(mask, 1,0)
            component_mask = ((components_target * mask) > 0).to(components_target.dtype)
        apply_mask_to = apply_mask_to*mask
        del mask
        torch.cuda.empty_cache()
        return component_mask, apply_mask_to
        
    
