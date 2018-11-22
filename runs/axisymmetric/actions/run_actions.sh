#!/bin/bash
#SBATCH --job-name=fix
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH -p cca 
#SBATCH --mem=0
#SBATCH -N1
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=fix_%j.log
pwd; hostname; date

source /mnt/home/abeane/.bash_profile

module load gcc
module load cuda
module load cudnn
module load openmpi

source /mnt/home/abeane/amuse_env/bin/activate

echo $CUDA_VISIBLE_DEVICES

export dir=`pwd`
cd ../../
source init.sh
cd $dir 

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

##mpirun -np 4 python axisymmetric_milkyway.py 
python -u compute_actions.py 

date
