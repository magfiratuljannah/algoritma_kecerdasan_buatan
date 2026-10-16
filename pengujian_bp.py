# PENGUJIAN BACK PROPAGATION - hanya propagasi maju (TIDAK ada backpropagation, bobot tidak berubah)
# Cara pakai:  python pengujian_bp.py gambar.jpg   atau   python pengujian_bp.py (pilih gambar lewat jendela)
# Notasi buku: x_i input, z_j hidden, y_k output, v_ij/v_0j (V), w_jk/w_0k (W)
import os, sys                               #os: file/folder, sys: argumen 
import numpy as np                           #numpy: hitungan matriks
from PIL import Image, ImageTk               #membaca & mengolah gambar, menampilkan jendela
import tkinter as tk                         #membuat jendela hasil pengujian
from tkinter import filedialog               #pemilihan file lewat jendela

f = lambda x: 1.0 / (1.0 + np.exp(-x))               # sigmoid biner (sama dengan training)

def praproses(path, ukuran):                          # sama persis dengan training
    asli = Image.open(path).convert("RGB")            # 1) gambar awal
    kecil = asli.resize((ukuran, ukuran), Image.LANCZOS)   # 2) resize dulu (warna)
    abu = kecil.convert("L")                          # 3) lalu grayscale
    return asli, kecil, abu, (np.asarray(abu, dtype=np.float64) / 255.0).reshape(-1)

def tampilkan(asli, kecil, abu, teks):                # jendela: gambar awal, resize, grayscale, hasil
    w = tk.Tk(); w.title("Hasil pengujian")
    a = asli.copy(); a.thumbnail((256, 256))          # gambar awal (diperkecil agar muat)
    big = lambda im: ImageTk.PhotoImage(im.resize((256, 256), Image.NEAREST))   # 32x32 diperbesar agar terlihat
    w.imgs = [ImageTk.PhotoImage(a), big(kecil), big(abu)]
    judul = [f"1. Gambar awal ({asli.size[0]}x{asli.size[1]})",
             f"2. Setelah resize ({kecil.size[0]}x{kecil.size[1]})",
             f"3. Setelah grayscale ({abu.size[0]}x{abu.size[1]})"]
    for c in range(3): # tampilkan 3 gambar di atas
        tk.Label(w, text=judul[c]).grid(row=0, column=c, padx=10, pady=5)
        tk.Label(w, image=w.imgs[c]).grid(row=1, column=c, padx=10)
    tk.Label(w, text=teks, font=("Arial", 12, "bold"), justify="left").grid(row=2, column=0, columnspan=3, pady=10)
    w.mainloop()

#main program: pilih gambar, baca bobot V & W, lalu propagasi maju (input -> hidden -> output)
if __name__ == "__main__":
    if len(sys.argv) > 1: path = sys.argv[1] #memilih gambar lewat argumen command line, misal: python pengujian_bp.py gambar.jpg
    else:
        r = tk.Tk(); r.withdraw()
        path = filedialog.askopenfilename(title="Pilih gambar pisang", filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp")])
        r.destroy()
    if not path or not os.path.isfile(path): sys.exit("Gambar tidak dipilih / tidak ditemukan.")
    if not os.path.isfile("bobot_vw.npz"): sys.exit("File 'bobot_vw.npz' tidak ada. Jalankan training_bp.py dulu.")

    d = np.load("bobot_vw.npz")                                       # baca bobot V dan W
    v_0j, v_ij, w_0k, w_jk = d["v_0j"], d["v_ij"], d["w_0k"], d["w_jk"]
    kelas = [str(k) for k in d["kelas"]]
    n, p = v_ij.shape; m = w_jk.shape[1]
    print(f"Arsitektur: 1 hidden layer | input X = {n} | hidden Z = {p} | output Y = {m}")

    asli, kecil, abu, x = praproses(path, int(d["UKURAN"]))
    z_j = f(v_0j + x @ v_ij)                                          # propagasi maju: input -> hidden
    y_k = f(w_0k + z_j @ w_jk)                                        # propagasi maju: hidden -> output

    k = int(np.argmax(y_k))                                           # output terbesar = kelas dikenali
    ok = y_k[k] >= 0.5 and np.all(np.delete(y_k, k) < 0.5)            # berhasil jika pemenang >= 0.5 & lainnya < 0.5
    out = "Output: " + "  ".join(f"Y{i+1}={y_k[i]:.4f}" for i in range(m))
    hasil = f"Hasil: gambar dikenali sebagai '{kelas[k]}' (Y{k+1} = {y_k[k]:.4f})"
    status = "STATUS: BERHASIL DIKENALI" if ok else "STATUS: TIDAK DIKENALI (output tidak jelas)"
    print(out); print(hasil); print(status)
    tampilkan(asli, kecil, abu, f"{out}\n{hasil}\n{status}")