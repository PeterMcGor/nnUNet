import pytest

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.run.run_training import run_training

plan_2d = "2d"
plan_3d_low = "3d_lowres"
plan_3d_full = "3d_fullres"
plan_3d_cascade = "3d_cascade_fullres"


dataset_id = "604"
fold = 1
plans_identifier = "nnUNetPlans"


@pytest.fixture
def setup_plans():
    preprocessed_dataset_folder_base = join(
        nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_id)
    )
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans = PlansManager(plans_file).plans["configurations"].keys()
    print("Available Plans", plans, type(plans))
    return plans


def test_3D_fullres_bloblossDS(setup_plans):
    if plan_3d_full not in setup_plans:
        pytest.skip(f"{plan_3d_full} is not an available plan")
    else:
        run_training(dataset_id, plan_3d_full, fold, "nnUNetTrainerBlobLossShort")


def test_2D_bloblossDS(setup_plans):
    if plan_2d not in setup_plans:
        pytest.skip(f"{plan_2d} is not an available plan")
    else:
        run_training(dataset_id, plan_2d, fold, "nnUNetTrainerBlobLossShort")


def test_3D_fullres_bloblossNoDS(setup_plans):
    if plan_3d_full in setup_plans:
        run_training(dataset_id, "3d_fullres", "0", "nnUNetTrainerBlobLossNoDSShort")
    else:
        pytest.skip(f"{plan_3d_full} is not an available plan")


def test_2D_bloblossNoDS(setup_plans):
    if plan_2d in setup_plans:
        run_training(dataset_id, plan_2d, fold, "nnUNetTrainerBlobLossNoDSShort")
    else:
        pytest.skip(f"{plan_2d} is not an available plan")


def test_3D_lowres_bloblossNoDS(setup_plans):
    if plan_3d_low in setup_plans:
        run_training(dataset_id, plan_3d_low, fold, "nnUNetTrainerBlobLossNoDSShort")
    else:
        pytest.skip(f"{plan_3d_low} is not an available plan")


def test_3D_lowres_blobloss(setup_plans):
    if plan_3d_low in setup_plans:
        run_training(dataset_id, plan_3d_low, fold, "nnUNetTrainerBlobLoss")
    else:
        pytest.skip(f"{plan_3d_low} is not an available plan")


def test_3D_cascade_blobloss(setup_plans):
    if plan_3d_cascade in setup_plans:
        run_training(dataset_id, plan_3d_cascade, "0", "nnUNetTrainerBlobLoss")
    else:
        pytest.skip(f"{plan_3d_cascade} is not an available plan")


def test_3D_cascade_bloblossNoDs(setup_plans):
    if plan_3d_cascade in setup_plans:
        run_training(dataset_id, plan_3d_cascade, "0", "nnUNetTrainerBlobLossNoDS")
    else:
        pytest.skip(f"{plan_3d_cascade} is not an available plan")
