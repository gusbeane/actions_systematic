#!/bin/bash
#SBATCH --job-name=acc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=8044322869@vtext.com
#SBATCH -p ib   
#SBATCH --oversubscribe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=52000
#SBATCH --time=02:00:00
#SBATCH --output=log/acc_%j.log
#SBATCH --array=0-1000
pwd; hostname; date

source /mnt/home/abeane/.bash_profile

export dir=`pwd`
cd ../
source init.sh
cd $dir

export OMP_NUM_THREADS=1 #this prevents conflicts in parallel libraries

##mpirun -np 4 python axisymmetric_milkyway.py 
echo $SLURM_ARRAY_TASK_ID
# /bin/time -f "-----\nreal %es\nuser %Us\nsys %Ss\nMaxRSS %Mk" python -u gen_movie_acc_map.py ../options ../cluster_snapshots.p $SLURM_ARRAY_TASK_ID
python -u gen_movie_acc_map.py ../options ../cluster_snapshots.p $SLURM_ARRAY_TASK_ID

date
