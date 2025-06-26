#!/usr/bin/env python3
import os
import sys

import numpy as np

import astropy.units as u
from astropy.io import fits
from lotaas_reprocessing import filterbank


if __name__ == "__main__":
    # Beam numbers
    central_beams = [i for i in range(13, 74)]
    outer_beams = [i for i in range(0, 12)]
    incoherent_beam = [12]
    all_beams = [i for i in range(0, 74)]

    obsid = sys.argv[1]

    path = "."

    # Compute mean beam
    if not os.path.exists(f"{path}/{obsid}_mean.fil"):
        print("Creating flatfield")
        fnames = [f"{path}/{obsid}_B{i:03d}.fil" for i in central_beams]
        nbeam = len(fnames)
        for i, fname in enumerate(fnames):
            print(fname)
            # Read filterbank
            fil = filterbank.FilterbankFile(fname, "read")
            data = fil.get_spectra(0, fil.nspec)
            fil.close()

            # Create
            if i == 0:
                mean_data = np.zeros_like(data)

            # Sum
            mean_data += data

        # Average
        nmean = nbeam
        mean_data /= nbeam

        # Write out mean
        outfname = f"{path}/{obsid}_mean.fil"
        outfil = filterbank.create_filterbank_file(outfname, fil.header, nbits=fil.nbits)
        outfil.append_spectra(mean_data)
        outfil.close()
    else:
        print("Reading flatfield")
        # Read filterbank
        fil = filterbank.FilterbankFile(f"{path}/{obsid}_mean.fil", "read")
        mean_data = fil.get_spectra(0, fil.nspec)
        fil.close()

    # Flat field all beams
    fnames = [f"{path}/{obsid}_B{i:03d}.fil" for i in all_beams]
    nbeam = len(fnames)
    for i, fname in enumerate(fnames):
        print(fname)
        # Read filterbank
        fil = filterbank.FilterbankFile(fname, "read")
        data = fil.get_spectra(0, fil.nspec)
        fil.close()
        
        data /= mean_data

        outfname = os.path.basename(fname).replace(".fil", "_ff.fil")
        outfil = filterbank.create_filterbank_file(outfname, fil.header, nbits=fil.nbits)
        outfil.append_spectra(data)
        outfil.close()