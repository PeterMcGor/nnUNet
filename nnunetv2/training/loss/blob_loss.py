import torch
from torch import nn, Tensor
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
import numpy as np

import SimpleITK as sitk

class BlobLoss(nn.Module):
    """
    """
    def __init__(self, global_loss_criterium: nn.Module, blob_loss_criterium: nn.Module, 
                 global_weight: float = 1.0, blob_weight: float = 2.0):
        super(BlobLoss, self).__init__()
        self.global_loss = global_loss_criterium
        self.blob_loss = blob_loss_criterium
        self.global_weight = global_weight
        self.blob_weight = blob_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        """
        # This is horrible but the framework doesnt give me many choices
        # TODO it is necessary to separate the batches
        training = target.shape[1] == 2
       
        if training: 
            labeled = target[:,1].unsqueeze(1)
            target = target[:,0].unsqueeze(1)
            #labeled_cpu = labeled.cpu()
            #print("Input", input.shape,"Target", target.shape,torch.unique(target), "Labeled", labeled.shape,torch.unique(labeled))
        else: 
            return self.global_loss(input, target.long())
        
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            #target = target[:, 0] 
            
        # Move tensors to CPU for intermediate calculations
        #input_cpu = input.cpu()
        

        global_loss = self.global_loss(input, target.long())
        #blob_loss  = torch.tensor([0], device=target.device) 
        blob_loss_per_sample = torch.zeros(target.shape[0], device=target.device)
        blob_loss_per_sample_atomic = [[] for i in range(target.shape[0])]
        #print("INIT", blob_loss_per_sample_atomic)
        with torch.no_grad():
            unique_components = torch.unique(labeled)
            ## Avoid dividing by zero
            unique_components_per_sample = torch.clamp(torch.tensor([torch.unique(sample).size()[0] for sample in labeled], device=target.device) - 1, min=1.0)
        #print("Components", unique_components,  unique_components_per_sample)
        n_comp = torch.tensor(unique_components.size()).to(labeled.device) - 1
        #print("Components", n_comp)
        #sitk.WriteImage(sitk.GetImageFromArray(labeled.cpu().numpy()), '/tmp/labels.nii.gz')
        #sitk.WriteImage(sitk.GetImageFromArray(input.cpu().detach().numpy().astype(np.float32)), '/tmp/input.nii.gz')
        input_detached = input.detach()
        input_detached.requires_grad = True
        dims = list(range(1,labeled.dim()))
        imag_id = np.random.randint(10000, 100000)
        for component in unique_components[1:]: # Avoid the bck
            #print("Comp", component)
            # assumes the first dimension as the batch size
            sample_contain_component = (labeled==component).sum(dim=dims)>0
            #print("Sample contain",component,sample_contain_component,  (labeled==component).sum(),(labeled==component).sum(dim=[1,2,3,4])>0)
            masked_target, masked_output = self.mask_all_labels(labeled[sample_contain_component], input_detached[sample_contain_component], label_to_keep=component)
            print(component, sample_contain_component,"Network output masked", masked_output.shape, "Masked Target", masked_target.shape, torch.unique(masked_target))
            b = 0
            blob_loss_1  = []
            for i,sample in enumerate(sample_contain_component):  
                if sample:
                    #sitk.WriteImage( sitk.GetImageFromArray(masked_target[b][0].detach().cpu().numpy()), '/tmp/id_'+str(imag_id)+'_comp_'+str(component.item())+'_b'+str(i)+str(masked_target[b].shape)+'.nii.gz')
                    #sitk.WriteImage( sitk.GetImageFromArray(masked_output[b][0].detach().cpu().numpy().astype(np.float32)), '/tmp/id_'+str(imag_id)+'_netOut_'+str(component.item())+'_b'+str(i)+str(masked_target[b].shape)+'.nii.gz')
                    blob_loss = self.blob_loss(masked_output[b].unsqueeze(0),  masked_target[b].unsqueeze(0)) 
                    blob_loss_1.append(blob_loss)
                    #blob_loss_per_sample[i] += blob_loss
                    #blob_loss_per_sample_atomic[b].append(blob_loss)
                    #print('blob_loss_per_sample_atomic', blob_loss_per_sample_atomic)
                    b+=1
                else:
                    pass
            print("blob_loss_1", blob_loss_1)
            blob_loss_1 = torch.mean(torch.stack([sample_loss for sample_loss in blob_loss_1])) if len(blob_loss_1)>0 else torch.tensor([0.0], device=target.device, requires_grad=True) * self.blob_weight
            var = input_detached.grad.sum() if input_detached.grad is not None else None
            print("masked_output grad", var, blob_loss.requires_grad, blob_loss_1)
            blob_loss_1.backward(inputs=input_detached)
            var = input_detached.grad.sum() if input_detached.grad is not None else None
            print("masked_output grad 2", var,blob_loss.requires_grad)
            #print()
            
            # TODO
            input.grad = input.grad +  input_detached.grad if input.grad is not None else input_detached.grad 
            
            
                    #blob_loss_per_sample[i] += 0
            #print("BLOBS",blob_loss_per_sample, unique_components_per_sample)
        #blob_loss_per_sample/=unique_components_per_sample
        #print("blob_loss_per_sample",blob_loss_per_sample)
        #blob_loss = blob_loss_per_sample.mean()
        #blob_loss_per_sample = [torch.mean(torch.stack(sample_losses)) for sample_losses in blob_loss_per_sample_atomic if len(sample_losses) >0]
        #print("blob_loss", blob_loss, blob_loss_per_sample)
        #blob_loss = torch.mean(torch.stack(blob_loss_per_sample)) if len(blob_loss_per_sample) > 0 else torch.tensor(0, dtype=global_loss.dtype, device=global_loss.device)
        #print("blob_loss", global_loss, global_loss.requires_grad, blob_loss, blob_loss.requires_grad)
        
                
            

            #torch.tensor([self.blob_loss(masked_output[b].unsqueeze(0),  masked_target[b].unsqueeze(0)) if sample==True else 0 for b,sample in enumerate(sample_contain_component)])
        #    blob_loss = blob_loss + self.blob_loss(masked_output,  masked_target)/n_comp
        #    del label_mask,  blob_loss_cpu
        #    torch.cuda.empty_cache()
        #return
        print()
        #print('loss', global_loss, print(global_loss.requires_grad), blob_loss,  print(blob_loss.requires_grad))
        #print()
        #+ self.blob_weight * blob_loss
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
        
    
    
class DummyLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.ones((1, ))
    
class RobustCrossEntropyLoss2(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # This is horrible but the framework doesnt give me many choices
        if target.shape[1] == 2: 
            labeled = target[:,1].unsqueeze(1)
            target = target[:,0].unsqueeze(1)
            print("Target", target.shape,torch.unique(target), "Labeled", labeled.shape,torch.unique(labeled))
        if len(target.shape) == len(input.shape):
            print("In len")
            assert target.shape[1] == 1
            target = target[:, 0] 
        print(torch.unique(target))
        global_loss = super().forward(input, target.long())
        blob_loss = 0
        return super().forward(input, target.long() )


class TopKLoss(RobustCrossEntropyLoss2):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

