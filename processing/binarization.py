import cv2
import os
import numpy as np
from tqdm import tqdm
from scipy.signal import wiener

# Parameters
P4 = 5    # Window size for Wiener filter
P3 = 500  # Minimum white area size to keep

#Updated input and output folders
input_folder = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\2_splitrgb_green"
output_folder = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\7_binarizedregion_uncropped"
os.makedirs(output_folder, exist_ok=True)

#Image list
#Automatically get all .jpg files from the input folder
image_names = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

for image_name in tqdm(image_names, desc="Processing Images"):
    input_path = os.path.join(input_folder, image_name)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        tqdm.write(f"❌ Could not read {image_name}. Skipping.")
        continue
    
    # Contrast Stretching
    p2, p98 = np.percentile(image, (3, 70))
    stretched = np.clip((image - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

    # Median Filtering
    median = cv2.medianBlur(stretched, 3)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0)
    clahe_img = clahe.apply(median)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(clahe_img, (3, 3), sigmaX=1)

    # Inversion
    inverted = cv2.bitwise_not(blurred)

    # Sobel Filtering
    sobelx = cv2.Sobel(inverted, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(inverted, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.clip(sobel_combined, 0, 255).astype(np.uint8)

    # Otsu Thresholding
    _, binary = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Opening (remove small white noise)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered = np.zeros_like(opened)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 300: # Adjust threshold as needed
            filtered[labels == i] = 255

    # Erosion to thin the structure
    eroded = cv2.erode(filtered, (2,2), iterations=1)

    # Save output
    output_name = image_name.replace(".jpg", "_final.jpg")
    output_path = os.path.join(output_folder, output_name)
    cv2.imwrite(output_path, eroded)
    tqdm.write(f"✅ Saved: {output_name}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

