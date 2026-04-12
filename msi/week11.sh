#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=16gb
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mcgon046@umn.edu
#SBATCH -p msismall
cd ~/psy8712-week11/msi
module load R/4.4.2-openblas-rocky8
Rscript week11-cluster.R