#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import imageio
import numpy as np
from skimage import measure
from stl.mesh import Mesh  # from numpy-stl

# --- Load TIFF stack as 3D volume ---
vol = np.array(imageio.mimread("P1-205CL-1_Void255.tif"))  # (nz, ny, nx)
print("Volume shape:", vol.shape, "dtype:", vol.dtype)

vmin, vmax = float(vol.min()), float(vol.max())
print("Min / Max:", vmin, vmax)

if vmin == vmax:
    raise ValueError("Volume has constant intensity; no surface can be extracted.")

# For your case: vmin=0, vmax=255, binary mask
# Choose iso-level strictly between them:
iso_val = 0.5 * (vmin + vmax)  # -> 127.5
print("Using iso_val:", iso_val)

# Convert to float for marching_cubes (safer)
vol_f = vol.astype(np.float32)

# --- Marching cubes to get surface ---
verts, faces, normals, _ = measure.marching_cubes(vol_f, level=iso_val,step_size=2,allow_degenerate=False)
print("Verts:", verts.shape, "Faces:", faces.shape)

# --- Build STL mesh ---
data = np.zeros(faces.shape[0], dtype=Mesh.dtype)
m = Mesh(data, remove_empty_areas=False)

for i, f in enumerate(faces):
    m.vectors[i] = verts[f]

out_name = "surface_binary_Void2.stl"
m.save(out_name)
print(f"Saved {out_name}")
