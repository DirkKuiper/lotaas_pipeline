# LOTAAS Reprocessing Pipeline

This repository contains a modular, SLURM-based pipeline for processing LOTAAS beams using Singularity containers. It handles staging, downloading, downsampling, flatfielding, GPU/CPU-based candidate generation, and post-processing.

## Directory Overview

```
lotaas_pipeline/

├── bin/                  # SLURM job scripts for each pipeline stage
├── containers/           # Singularity image and definition
├── lotaas_reprocessing/  # Core modules
├── preproc/              # Preprocessing scripts
├── pipeline/             # Core pipeline stages
├── postproc/             # Post-detection classification and plotting
├── staging/              # Python staging script + temporary staging files
├── logs/                 # Master logs for each pipeline run
├── master_pipeline.slurm # Master orchestration script
├── settings.yaml         # Shared config
└── graveyard/            # Old or unused files
```

## Usage

### 1. Prepare input

Create a plain-text list of SRM URLs for the beams you want to process:

```
srm_list.txt
```

Each line should be a full SRM URL (e.g., starting with `srm://srm.grid.sara.nl/...`).

### 2. Submit the full pipeline (automatic staging)

To run the full pipeline including automatic staging via the LTA API:

```bash
sbatch master_pipeline.slurm srm_list.txt
```

This will:

- Create a dedicated staging directory (with `webdav_links.txt` and `macaroon.txt`)
- Clean and sort the WebDAV input file
- Submit job arrays for downloading and downsampling
- Launch sequential jobs for flatfielding and full GPU+CPU processing
- Move all logs into a permanent location within the data directory
- Clean up the temporary staging folder

**For more information on the staging API, see the [`staging/README.md`](staging/README.md).**

### Optional: Skip staging if already complete

If the data is already staged and you have the WebDAV links and macaroon token:

```bash
sbatch master_pipeline.slurm webdav_links.txt "<macaroon_token>"
```

Note:
- `webdav_links.txt` should be the file containing the staged WebDAV download URLs (one per line)
- The second argument is the macaroon token provided as a raw string (enclosed in quotes)

The pipeline will detect the presence of two arguments and skip the staging step accordingly.
