#!/bin/bash
#SBATCH --job-name=amuse
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -n 8 
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200g
#SBATCH --time=7-00:00:00
#SBATCH --output=log/asmw_%j.log
#SBATCH --constraint=p100
pwd; hostname; date

source /mnt/home/abeane/.bash_profile

module load gcc
module load cuda
module load cudnn
module load openmpi

source /mnt/home/abeane/amuse_env/bin/activate
export PYTHONPATH=

echo $CUDA_VISIBLE_DEVICES

export dir=`pwd`
cd ../
source init.sh
cd $dir

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

##mpirun -np 4 python axisymmetric_milkyway.py 
python -u oc_nbody.py options 

date
