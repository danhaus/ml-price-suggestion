#!/bin/bash

module load conda/py3-latest

#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=10:00:00          # walltime

python bag_of_words_random_forest.py