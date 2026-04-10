import nibabel as nib
import numpy as np
import os
from PIL import Image

def extract_slices(nifti_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    img = nib.load(nifti_path)
    volume = img.get_fdata()

    saved = 0
    for i in range(volume.shape[2]):
        slice_i = volume[:, :, i]

        # Skip empty slices
        if np.mean(slice_i) < 10:
            continue

        # Normalize to 0-255
        slice_i = (slice_i - np.min(slice_i)) / (np.max(slice_i) - np.min(slice_i))
        slice_i = (slice_i * 255).astype(np.uint8)

        img_pil = Image.fromarray(slice_i)
        img_pil.save(os.path.join(output_folder, f"slice_{i:03d}.png"))
        saved += 1

    print(f"Saved {saved} slices → {output_folder}")