#!/bin/bash
#SBATCH --job-name=gpu_job_test
#SBATCH --chdir=/home/pe1344
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pedro.maciasgordaliza@unil.ch
#SBATCH --ntasks=1
#SBATCH --time=02:40:00
#SBATCH --output=GPU_test_out_%N.%j.%a.out
#SBATCH --error=GPU_test_err_%N.%j.%a.err
## Apparently these lines are needed for GPU execution
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --gres=gpu:1

singularity run --bind /home/{youruser}/nnUNet_test_data:/opt/nnunet_resources  --nv docker://petermcgor/nnunet:0.0.1 nnUNet_train 2d nnUNetTrainerV2 Task002_Heart 1 --npz
