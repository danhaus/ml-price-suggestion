#!/bin/bash

#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=24:00:00          # walltime

module load conda/py3-latest

source activate IP
pip install pandas

python bag_of_words_random_forest.py