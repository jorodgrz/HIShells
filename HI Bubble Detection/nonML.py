import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from scipy.ndimage import gaussian_filter, label
from skimage import measure, morphology
from skimage.draw import polygon_perimeter

# === Load and clean ===
filename = 'NGC_2403_NA_MOM0_THINGS.FITS'  # your THINGS Moment 0 map
hdu = fits.open(filename)[0]
data = hdu.data.squeeze()
data = np.nan_to_num(data, nan=1e-3)

# === Difference of Gaussians (DoG) ===
smooth_small = gaussian_filter(data, sigma=1)
smooth_large = gaussian_filter(data, sigma=4)
dog = smooth_small - smooth_large  # Highlights local depressions

# === Threshold DoG map for depressions ===
threshold = np.percentile(dog, 5)  # bottom 5% = local minima
mask = dog < threshold
mask = morphology.remove_small_objects(mask, min_size=80)
mask = morphology.binary_closing(mask, morphology.disk(2))

# === Label and analyze ===
labeled, _ = label(mask)
regions = measure.regionprops(labeled)

# === Galactic disk mask (optional) ===
yc, xc = data.shape[0] // 2, data.shape[1] // 2
Y, X = np.ogrid[:data.shape[0], :data.shape[1]]
r = np.sqrt((X - xc)**2 + (Y - yc)**2)
r_mask = r < 1000

# === Filtering parameters ===
min_area = 80
max_area = 4000
min_circularity = 0.25
min_solidity = 0.5

shells = []
for region in regions:
    cy, cx = int(region.centroid[0]), int(region.centroid[1])
    if not r_mask[cy, cx]:
        continue
    if region.area < min_area or region.area > max_area or region.perimeter == 0:
        continue
    circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
    if circularity < min_circularity or region.solidity < min_solidity:
        continue
    region._score = circularity * region.solidity * region.area * np.exp(-r[cy, cx] / 500)
    shells.append(region)

# === Sort and plot ===
shells.sort(key=lambda r: r._score, reverse=True)
N = 600

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
norm = simple_norm(data, 'sqrt')

# Original HI with shell overlays
ax[0].imshow(data, origin='lower', cmap='inferno', norm=norm)
for region in shells[:N]:
    rr, cc = polygon_perimeter(region.coords[:, 0], region.coords[:, 1], shape=data.shape, clip=True)
    ax[0].plot(cc, rr, color='cyan', linewidth=1)
ax[0].set_title(f'Top {N} HI Shells via DoG (of {len(shells)} total)')

# DoG depression map
ax[1].imshow(dog, origin='lower', cmap='coolwarm', vmin=np.percentile(dog, 1), vmax=np.percentile(dog, 99))
ax[1].contour(mask, levels=[0.5], colors='white', linewidths=0.5)
ax[1].set_title('DoG: Local Depression Map')
plt.tight_layout()
plt.show()