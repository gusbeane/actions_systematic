#!/bin/bash
#SBATCH --job-name=asmw
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH -p cca
#SBATCH --mem=0
#SBATCH -N1 --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=asmw_%j.log
#SBATCH --constraint=skylake
pwd; hostname; date

source /mnt/home/abeane/.bash_profile
source /mnt/home/abeane/amuse_env/bin/activate

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

##mpirun -np 4 python axisymmetric_milkyway.py 
python compute_schmactions.py

date
