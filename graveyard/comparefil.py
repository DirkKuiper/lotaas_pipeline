# import numpy as np
# import argparse
# from lotaas_reprocessing import filterbank  # Ensure your filterbank module is correctly installed
# import matplotlib.pyplot as plt

# def load_filterbank(filename):
#     """Load a filterbank file and return its metadata and data"""
#     fil = filterbank.FilterbankFile(filename)
#     data = fil.get_spectra(0, fil.nspec)  # Load full spectrum data
#     return fil, data

# def compare_headers(fil1, fil2):
#     """Compare the metadata (headers) of two filterbank files"""
#     print("\n🔍 Comparing Headers:\n")
#     keys1 = set(fil1.header.keys())
#     keys2 = set(fil2.header.keys())

#     common_keys = keys1.intersection(keys2)
#     for key in sorted(common_keys):
#         val1 = fil1.header[key]
#         val2 = fil2.header[key]
#         if val1 != val2:
#             print(f"❌ {key}: {val1} -> {val2}")

#     # Check if there are missing keys
#     for key in keys1 - keys2:
#         print(f"⚠️ Missing in File 2: {key} -> {fil1.header[key]}")
#     for key in keys2 - keys1:
#         print(f"⚠️ Missing in File 1: {key} -> {fil2.header[key]}")

# def compare_data(data1, data2):
#     """Compare the actual data from two filterbank files"""
#     print("\n🔍 Comparing Data:\n")
#     max_diff = np.max(np.abs(data1 - data2))
#     mean_diff = np.mean(np.abs(data1 - data2))
#     print(f"🔹 Max Difference: {max_diff}")
#     print(f"🔹 Mean Difference: {mean_diff}")

#     # If data is different, plot the differences
#     if max_diff > 1e-6:
#         plt.figure(figsize=(10, 5))
#         plt.imshow(np.abs(data1 - data2), aspect="auto", cmap="viridis")
#         plt.colorbar(label="Difference")
#         plt.xlabel("Time Sample")
#         plt.ylabel("Frequency Channel")
#         plt.title("Difference Between Two Outputs")
#         plt.show()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compare two filterbank files")
#     parser.add_argument("file1", help="First filterbank file")
#     parser.add_argument("file2", help="Second filterbank file")
#     args = parser.parse_args()

#     # Load files
#     fil1, data1 = load_filterbank(args.file1)
#     fil2, data2 = load_filterbank(args.file2)

#     # Compare metadata headers
#     compare_headers(fil1, fil2)

#     # Compare actual data
#     compare_data(data1, data2)

from astropy.io import fits

# Open the FITS file
hdu = fits.open("new_output2.fits")

# Print SUBINT header
print("\n🔍 SUBINT Header:")
print(hdu['SUBINT'].header)

# Check if DAT_FREQ exists
if 'DAT_FREQ' in hdu['SUBINT'].data.names:
    print("\n🔍 DAT_FREQ values (first subint):")
    print(hdu['SUBINT'].data['DAT_FREQ'][0])  # Print the first subint's frequency array
else:
    print("\n⚠️ DAT_FREQ not found in the file!")

# Close the file
hdu.close()