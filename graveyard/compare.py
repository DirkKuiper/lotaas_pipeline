from astropy.io import fits

def compare_fits_headers(original_file, downsampled_file):
    # Open original and downsampled PSRFITS
    original_hdu = fits.open(original_file)
    downsampled_hdu = fits.open(downsampled_file)

    # Compare primary headers
    print("\n🔍 Comparing Primary Header:")
    for key in original_hdu[0].header.keys():
        if key in downsampled_hdu[0].header:
            if original_hdu[0].header[key] != downsampled_hdu[0].header[key]:
                print(f"❌ {key}: {original_hdu[0].header[key]} -> {downsampled_hdu[0].header[key]}")
        else:
            print(f"⚠️ Missing key in downsampled file: {key}")

    # Compare SUBINT headers
    print("\n🔍 Comparing SUBINT Header:")
    for key in original_hdu['SUBINT'].header.keys():
        if key in downsampled_hdu['SUBINT'].header:
            if original_hdu['SUBINT'].header[key] != downsampled_hdu['SUBINT'].header[key]:
                print(f"❌ {key}: {original_hdu['SUBINT'].header[key]} -> {downsampled_hdu['SUBINT'].header[key]}")
        else:
            print(f"⚠️ Missing key in downsampled file: {key}")

    original_hdu.close()
    downsampled_hdu.close()

# Example usage
compare_fits_headers("L652970_SAP0_BEAM6_2bit.fits", "new_output.fits")