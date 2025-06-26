# LOTAAS Reprocessing Pipeline

This repository contains a modular, SLURM-based pipeline for processing LOTAAS beams using Singularity containers. It handles downloading, downsampling, flatfielding, GPU/CPU-based candidate generation, and post-processing.

## Directory Overview

lotaas_reprocessing/

├── bin/                  # SLURM job scripts for each pipeline stage

├── containers/           # Singularity image and definition

├── preproc/              # Preprocessing scripts

├── pipeline/             # Core pipeline stages

├── postproc/             # Post-detection classification and plotting

├── lotaas_reprocessing/  # Core Python package modules

├── master.sh             # Master orchestration script

├── settings.yaml         # Shared config

├── graveyard/            # Old or unused files


## Usage

### 1. Prepare input

After staging beams via the LOFAR LTA StageIt system, download:
- The WebDAV URL list (e.g., `L1268206.txt`)
- The associated macaroon file (e.g., `L1268206.macaroon`)

Place both in the project root.

### 2. Run the full pipeline

Submit the master orchestration script:

`sbatch master.sh L1268206.txt L1268206.macaroon`

This will:
- Clean the input file
- Submit the download job array
- Launch sequential jobs for downsampling, flatfielding, GPU and CPU processing
- Organize all logs by OBSID and SAP