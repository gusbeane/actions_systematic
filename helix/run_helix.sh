#!/bin/bash
#SBATCH --job-name=helix
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH -p cca
#SBATCH --ntasks-per-node=40
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=helix_%j.log
#SBATCH --constraint=skylake
#SBATCH --array=0-2
pwd; hostname; date

source /mnt/home/abeane/.bash_profile

echo $CUDA_VISIBLE_DEVICES

export dir=`pwd`
cd ../
source init.sh
cd $dir

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

python -u helix.py $SLURM_ARRAY_TASK_ID

date
