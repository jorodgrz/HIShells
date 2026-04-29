from astropy.io import fits

path = "Data/DDO154_NA_CUBE_THINGS.fits"

with fits.open(path) as hdul:
    hdul.info()                  # summary of all HDUs
    hdr  = hdul[0].header        # primary header (a dict-like object)
    data = hdul[0].data          # numpy ndarray of the cube

print(repr(hdr))                 # pretty-print full header

print("data.shape =", data.shape)        # numpy axis order: (NAXIS4, NAXIS3, NAXIS2, NAXIS1)
print("data.dtype =", data.dtype)

print("NAXIS  =", hdr["NAXIS"])
print("NAXIS1 =", hdr["NAXIS1"], "->", hdr["CTYPE1"])  # columns (RA)
print("NAXIS2 =", hdr["NAXIS2"], "->", hdr["CTYPE2"])  # rows (Dec)
print("NAXIS3 =", hdr["NAXIS3"], "->", hdr["CTYPE3"])  # velocity channels
print("NAXIS4 =", hdr["NAXIS4"], "->", hdr["CTYPE4"])  # Stokes
print("BUNIT  =", hdr["BUNIT"])