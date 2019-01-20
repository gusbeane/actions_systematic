#!/bin/bash
#SBATCH --job-name=orbit
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH -p bnl 
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --output=log_%j.log
pwd; hostname; date

source /mnt/home/abeane/.bash_profile

echo $CUDA_VISIBLE_DEVICES

export dir=`pwd`
cd ../../
source init.sh
cd $dir

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

python -u calc_many_orbits.py

date
