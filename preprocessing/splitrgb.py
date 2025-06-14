from PIL import Image
import os
from tqdm import tqdm

def split_rgb_channels_with_progress(image_paths, output_dir):
    """
    Memuat gambar, memisahkan ke saluran RGB, dan menyimpan setiap saluran sebagai gambar terpisah
    dengan progress bar.

    Args:
        image_paths (list): Daftar path lengkap ke file gambar.
        output_dir (str): Direktori untuk menyimpan gambar saluran yang terpisah.
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Memproses Gambar"):
        try:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(img_path)

            if img.mode != 'RGB':
                img = img.convert('RGB')

            r, g, b = Image.Image.split(img)

            # Simpan dengan format xxx_red.jpg
            r.save(os.path.join(output_dir, f"{base_name}_red.jpg"), quality=100)
            g.save(os.path.join(output_dir, f"{base_name}_green.jpg"), quality=100)
            b.save(os.path.join(output_dir, f"{base_name}_blue.jpg"), quality=100)

        except FileNotFoundError:
            tqdm.write(f"Error: File tidak ditemukan di {img_path}")
        except Exception as e:
            tqdm.write(f"Terjadi kesalahan saat memproses '{img_path}': {e}")

# Path direktori input dan output
input_dir = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\1_inputimagesnew"
output_dir = r"C:\Users\Zenbook\Documents\tubespcb[1]\data\2_splitrgb"

# Buat daftar file gambar yang ingin diproses (01.jpg hingga 33.jpg)
image_names = [f"{i:02d}.jpg" for i in range(1, 34)]
image_paths = [os.path.join(input_dir, name) for name in image_names]

# Jalankan fungsi
split_rgb_channels_with_progress(image_paths, output_dir)
