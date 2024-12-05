import torch

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerBlobLoss import nnUNetTrainerBlobLoss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerBlobLossNoDS import nnUNetTrainerBlobLossNoDS

class nnUNetTrainerBlobLossNoDSShort(nnUNetTrainerBlobLossNoDS):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.batch_size = 2
        self.num_iterations_per_epoch = 2
        self.num_epochs = 2
        self.num_val_iterations_per_epoch = 2
        
class nnUNetTrainerBlobLossShort(nnUNetTrainerBlobLoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        print(self.batch_size, self.num_iterations_per_epoch, self.num_epochs, self.num_val_iterations_per_epoch )
        self.batch_size = 2
        self.num_iterations_per_epoch = 250
        self.num_epochs = 1000
        self.num_val_iterations_per_epoch = 50 
        
        