import cv2
import os
import numpy as np
import csv
from tqdm import tqdm

input_folder = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\7_binarizedregion_uncropped"
output_csv = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\white_pixel_count_uncropped.csv"

# Automatically get all .jpg files from the input folder
image_names = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

# Prepare CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "White Pixel Count"])

    # Process each image
    for image_name in tqdm(image_names, desc="Processing Images"):
        input_path = os.path.join(input_folder, image_name)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            tqdm.write(f"❌ Could not read {image_name}. Skipping.")
            continue

        # Count white pixels
        white_pixel_count = np.sum(image == 255)

        # Write to CSV
        writer.writerow([image_name, white_pixel_count])

print(f"✅ White pixel counts saved to {output_csv}")

