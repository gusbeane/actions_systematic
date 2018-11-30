#!/bin/bash
#SBATCH --job-name=amuse
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -n 8 
#SBATCH --ntasks-per-node=8
#SBATCH --mem=150g

###SBATCH -p cca
###SBATCH -N1
###SBATCH --ntasks-per-node=40
###SBATCH --mem=0
###SBATCH --constraint=skylake

#SBATCH --time=7-00:00:00
#SBATCH --output=log/asmw_%j.log
pwd; hostname; date

source /mnt/home/abeane/.bash_profile

echo $CUDA_VISIBLE_DEVICES

export dir=`pwd`
cd ../
source init.sh
cd $dir

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

python -u oc_nbody.py options_m12i_long

date
