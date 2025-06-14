import cv2
import numpy as np
import os
from tqdm import tqdm

# Folder path
folder = r"D:\Coolyeah\Semester_10_terakhir\03_PCB\Tubes2\data\3_inpaintingmask"

# Channels to process
channels = ['r', 'g', 'b']

# Process each channel mask for image 001 to 008
for i in tqdm(range(1, 9), desc="Processing all channel masks"):
    image_id = f"{i:03d}"
    for ch in channels:
        input_filename = f"{image_id}_{ch}mask.jpg"
        output_filename = f"{image_id}_{ch}mask2.jpg"

        input_path = os.path.join(folder, input_filename)
        output_path = os.path.join(folder, output_filename)

        # Read grayscale mask
        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            tqdm.write(f"Error: Cannot read image {input_path}")
            continue

        # Structuring element for dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        # Dilate the mask
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # Apply Gaussian smoothing
        smoothed = cv2.GaussianBlur(dilated, (5, 5), 0)

        # Save the result
        cv2.imwrite(output_path, smoothed)

print("All red, green, and blue masks processed and saved with '_mask2'.")
