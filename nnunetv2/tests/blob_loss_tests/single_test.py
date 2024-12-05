# run_single_test.py
import os
os.environ["nnUNet_raw"] = "/code/nnUNet2/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/code/nnUNet2/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/code/nnUNet2/nnUNet_results"
dataset_id = "824"
fold = 0
plans_identifier = "nnUNetPlans"
plan_2d = "2d"
plan_3d_low = "3d_lowres"
plan_3d_full = "3d_fullres"
plan_3d_cascade = "3d_cascade_fullres"
  
import pytest
from nnunetv2.paths import nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.run.run_training import run_training

def setup_plans():
    print(nnUNet_preprocessed)
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans = PlansManager(plans_file).plans["configurations"].keys()
    print("Available Plans", plans, type(plans))
    return plans


def run_single_test():
    setup_plan  = setup_plans()
    if plan_3d_full not in setup_plan:
        pytest.skip(f"{plan_3d_full} is not an available plan")
    else:
        run_training(dataset_id, plan_3d_full, fold, "nnUNetTrainerBlobLossShort")

if __name__ == "__main__":
    # Specify the test file and function you want to run
    test_file = 'blob_loss_test.py'
    test_function = 'test_3D_fullres_bloblossNoDS'

    # Run the specified test function with the debugger
    run_single_test()