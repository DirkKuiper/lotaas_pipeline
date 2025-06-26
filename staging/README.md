# LOFAR LTA Staging API

This directory contains tools for automating the staging of LOFAR LTA (Long-Term Archive) data via the StageIt API. It enables bulk staging of SRM URLs and automatic extraction of WebDAV download links and macaroon tokens for downstream processing. The scripts have been adapted from code developed by [ASTRON](https://git.astron.nl/astron-sdc/lofar_stager_api).

---

## Files

- `stage_and_extract.py` – Main script for submitting SRM staging requests and waiting for completion.
- `stager_proxy.py`, `stager_access.py` – API interface modules used internally by the staging script.
- `.stagingrc` – Required user credentials file for authenticating with the StageIt system.

---

## Usage

### 1. Prepare SRM List

Create a text file (e.g., `srm_list.txt`) containing one SRM URL per line:

```
srm://srm.grid.sara.nl:8443/pnfs/grid.sara.nl/data/lofar/...
srm://srm.grid.sara.nl:8443/pnfs/grid.sara.nl/data/lofar/...
```

### 2. Run Staging

```bash
python3 staging/stage_and_extract.py srm_list.txt output_directory/
```

This will:
- Submit a staging request to the LTA
- Poll the request until it completes
- Save the WebDAV links to `webdav_links.txt`
- Save the macaroon token to `macaroon.txt`

---

## Required: Authentication File

Make sure to create a `.stagingrc` file in the `staging/` directory with the following format:



This file is required for the staging script to authenticate and communicate with the StageIt API.