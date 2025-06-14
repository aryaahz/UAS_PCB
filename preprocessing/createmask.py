import cv2
import os
import numpy as np
from tqdm import tqdm

# Input and output folders
input_folder = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\2_splitrgb_green"
output_folder = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\3_inpaintingmask"
os.makedirs(output_folder, exist_ok=True)

# Threshold mapping per image number
threshold_map = {
    '01': 240,
    '02': 240,
    '03': 220,
    '04': 220,
    '05': 230,
    '06': 240,
    '07': 240,
    '08': 240
}

channels = ['red', 'green', 'blue']
numbers = list(threshold_map.keys())

# Process each image
for number in tqdm(numbers, desc="Processing masks"):
    threshold_value = threshold_map[number]
    for color in channels:
        input_filename = f"{number}_{color}.jpg"
        input_path = os.path.join(input_folder, input_filename)

        # Load image in grayscale
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            tqdm.write(f"Warning: Cannot load image {input_path}")
            continue

        # Apply threshold
        _, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

        # Create output filename
        suffix = f"{color[0]}mask.jpg"  # rmask, gmask, bmask
        output_filename = f"{number}_{suffix}"
        output_path = os.path.join(output_folder, output_filename)

        # Save binary mask
        cv2.imwrite(output_path, binary_mask)

print("All custom-threshold masks processed and saved.")
