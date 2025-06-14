import cv2
import os

# Path ke gambar input
image_path = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\2_splitrgb_green\26.jpg"
image = cv2.imread(image_path)

# Cek apakah gambar berhasil dibuka
if image is None:
    print("❌ Gambar tidak ditemukan. Cek kembali path-nya:")
    print(image_path)
else:
    # Tampilkan window untuk cropping interaktif
    roi = cv2.selectROI("Pilih area lalu tekan ENTER atau SPACE", image, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi

    # Crop gambar
    cropped_image = image[y:y+h, x:x+w]

    # Path untuk menyimpan gambar hasil crop
    output_folder = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\5_croppedregion"
    os.makedirs(output_folder, exist_ok=True)  # Buat folder jika belum ada
    output_path = os.path.join(output_folder, "26_1.jpg")

    # Simpan gambar hasil crop
    cv2.imwrite(output_path, cropped_image)
    print(f"✅ Gambar hasil crop berhasil disimpan di: {output_path}")

    # Tampilkan hasil crop
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
