#!/bin/sh
# Moab-PBS staging file
#PBS -N rl
#PBS -q rudens
##PBS -l walltime=24:00:00, nodes=1:ppn=4:gpus=2
#PBS -l nodes=1:ppn=2:gpus=1,feature=centos7,walltime=96:00:00
#PBS -j oe
##PBS -t 1-10


module load cuda/cuda-7.5
export TMPDIR=$HOME/tmp
source $HOME/machinelearn/bin/activate
cd $HOME/rl/
python main.py




