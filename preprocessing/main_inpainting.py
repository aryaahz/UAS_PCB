import cv2
import os
from Inpainter import Inpainter
from tqdm import tqdm

# Base folder paths
input_rgb_folder = r"D:\Coolyeah\Semester_10_terakhir\03_PCB\Tubes2\data\1_inputimages"
input_green_folder = r"D:\Coolyeah\Semester_10_terakhir\03_PCB\Tubes2\data\2_splitrgb"
mask_folder = r"D:\Coolyeah\Semester_10_terakhir\03_PCB\Tubes2\data\3_inpaintingmask"
output_folder = r"D:\Coolyeah\Semester_10_terakhir\03_PCB\Tubes2\data\4_readytouse"
os.makedirs(output_folder, exist_ok=True)

# Image IDs to process
image_ids = [f"{i:03d}" for i in range(1, 9)]  # 001 to 008

# Half patch width for inpainting
halfPatchWidth = 7

def inpaint_image(image_path, mask_path, output_path):
    originalImage = cv2.imread(image_path, 1)
    inpaintMask = cv2.imread(mask_path, 0)

    if originalImage is None:
        tqdm.write(f"❌ Cannot read image: {image_path}")
        return
    if inpaintMask is None:
        tqdm.write(f"❌ Cannot read mask: {mask_path}")
        return

    i = Inpainter(originalImage, inpaintMask, halfPatchWidth)
    if i.checkValidInputs() == i.CHECK_VALID:
        i.inpaint()
        cv2.imwrite(output_path, i.result)
    else:
        tqdm.write(f"❌ Invalid input: {image_path}")

def main():
    for img_id in tqdm(image_ids, desc="Inpainting all images"):
        mask_path = os.path.join(mask_folder, f"{img_id}_gmask2.jpg")

        # Inpaint RGB
        rgb_path = os.path.join(input_rgb_folder, f"{img_id}.jpg")
        rgb_output_path = os.path.join(output_folder, f"{img_id}_inpainted_rgb.jpg")
        inpaint_image(rgb_path, mask_path, rgb_output_path)

        # Inpaint Green channel
        green_path = os.path.join(input_green_folder, f"{img_id}_green.jpg")
        green_output_path = os.path.join(output_folder, f"{img_id}_inpainted_green.jpg")
        inpaint_image(green_path, mask_path, green_output_path)

    print("✅ All inpainting complete.")

if __name__ == "__main__":
    main()
