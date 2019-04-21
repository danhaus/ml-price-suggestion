#!/bin/bash

#SBATCH --ntasks-per-node=28     # Tasks per node
#SBATCH --nodes=2                # Number of nodes requested
#SBATCH --time=20:00:00          # walltime

module load conda/py3-latest

python principal_embeddings_random_forest.py