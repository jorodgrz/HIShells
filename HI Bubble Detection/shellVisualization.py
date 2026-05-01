import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# === CONFIGURATION ===
GALAXY_NAME = "NGC 2403"
TABLE7_PATH = "table7.dat"
TABLE2_PATH = "table2.dat"
OUTPUT_REGION_FILE = f"{GALAXY_NAME}_shells.reg"

# === STEP 1: Load and filter table7 ===
colspecs_7 = [(0, 11), (12, 15), (16, 18), (19, 21), (22, 26), (27, 28), (28, 30),
              (31, 33), (34, 38), (39, 43), (44, 45), (46, 50), (51, 53), (54, 57),
              (58, 61), (62, 66), (67, 71), (72, 75), (76, 80), (81, 85)]
names_7 = ['Name', 'Seq', 'RAh', 'RAm', 'RAs', 'DEsign', 'DEd', 'DEm', 'DEs',
           'HV', 'Type', 'Diameter', 'Vexp', 'PA', 'AxialRatio', 'Rgal', 'nHI', 'tkin', 'logE', 'logMHI']

df = pd.read_fwf(TABLE7_PATH, colspecs=colspecs_7, names=names_7)
df = df[df['Name'].str.strip() == GALAXY_NAME]

# === STEP 2: Get galaxy distance from table2 ===
colspecs_2 = [(0, 11), (12, 21), (22, 24), (25, 27), (28, 32), (33, 34), (34, 36),
              (37, 39), (40, 44), (45, 55), (56, 60), (61, 63), (64, 67),
              (68, 72), (73, 78), (80, 84), (85, 88)]
names_2 = ['Name', 'OName', 'RAh', 'RAm', 'RAs', 'DEsign', 'DEd', 'DEm', 'DEs',
           'Type', 'Dist', 'Incl', 'PA', 'MHI', 'logSFR', 'logD25', 'Res']
df2 = pd.read_fwf(TABLE2_PATH, colspecs=colspecs_2, names=names_2)
dist = df2[df2['Name'].str.strip() == GALAXY_NAME]['Dist'].values[0]

# === STEP 3: Generate region strings ===
region_lines = [
    "# Region file format: DS9",
    "global color=red font=\"helvetica 10 bold\"",
    "fk5"  # <-- ADD THIS LINE
]

for i, row in df.iterrows():
    try:
        # RA/Dec formatting
        ra_str = f"{int(row['RAh'])}h{int(row['RAm'])}m{float(row['RAs']):.2f}s"
        dec_str = f"{'-' if row['DEsign'] == '-' else '+'}{int(row['DEd'])}d{int(row['DEm'])}m{float(row['DEs']):.2f}s"
        coord = SkyCoord(ra=ra_str, dec=dec_str, frame='icrs')

        # Parse diameter
        diameter_pc = float(row['Diameter'])
        if diameter_pc <= 0:
            print(f"[!] Skipping: Diameter {diameter_pc} is invalid (row {i})")
            continue

        # Convert diameter to arcsec
        diameter_arcsec = (diameter_pc / (dist * 1e6)) * 206265

        # Axial ratio and PA
        axial_ratio = float(row['AxialRatio']) if pd.notnull(row['AxialRatio']) else 1.0
        if axial_ratio <= 0:
            print(f"[!] Skipping: Axial ratio {axial_ratio} is invalid (row {i})")
            continue

        pa = float(row['PA']) if pd.notnull(row['PA']) else 0.0

        # Compute ellipse dimensions
        major_axis = diameter_arcsec
        minor_axis = diameter_arcsec * axial_ratio

        if major_axis <= 0 or minor_axis <= 0:
            print(f"[!] Skipping: Invalid ellipse size (row {i})")
            continue

        region_line = f"ellipse({coord.ra.degree:.6f},{coord.dec.degree:.6f},{major_axis/2:.2f}\",{minor_axis/2:.2f}\",{pa:.1f})"
        print(f"[{i}] RA={coord.ra.degree:.6f}, Dec={coord.dec.degree:.6f}, "
        f"Major={major_axis/2:.2f}\" Arcsec, Minor={minor_axis/2:.2f}\" Arcsec, PA={pa:.1f}")
        region_lines.append(region_line)

    except Exception as e:
        print(f"[!] Skipped row {i} due to error: {e}")

# === STEP 4: Write to .reg file ===
with open(OUTPUT_REGION_FILE, "w") as f:
    for line in region_lines:
        f.write(line + "\n")

print(f"[✓] Region file saved to: {OUTPUT_REGION_FILE}")