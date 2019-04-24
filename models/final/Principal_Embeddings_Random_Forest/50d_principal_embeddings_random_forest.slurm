#!/bin/bash

#SBATCH --ntasks-per-node=28     # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=24:00:00          # walltime

module load conda/py3-latest

python 50d_principal_embeddings_random_forest.py