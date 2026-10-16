# TRAINING BACK PROPAGATION - klasifikasi pisang
# Notasi buku: x_i input, z_j hidden, y_k output, t_k target, v_ij/v_0j bobot & bias
# input->hidden, w_jk/w_0k bobot & bias hidden->output, alpha learning rate.
# Output: bobot_vw.npz (berisi V dan W)
import os, sys                          # os: file/folder, sys: argumen
import numpy as np                      # numpy: hitungan matriks
from PIL import Image                   # PIL: baca & olah gambar
  
FOLDER = "buah_pisang"                  # folder dataset (train, valid, test)
UKURAN = 32                             # semua gambar dijadikan 32 x 32
n, p, m = UKURAN * UKURAN, 30, 3        # n input = 1024 pixel, p hidden = 30, m output = 3
alpha, EPOCH_MAKS, TARGET = 0.1, 500, 0.001 #alpha = learning rate, EPOCH_MAKS = epoch maksimum, TARGET = error minimum
KELAS = {"belum matang": [1, 0, 0], "matang": [0, 1, 0], "busuk": [0, 0, 1]}   # kelas -> target

f  = lambda x: 1.0 / (1.0 + np.exp(-x))     #ubah angka apa pun jadi antara 0 dan 1
df = lambda fx: fx * (1.0 - fx)             #menetukan seberapa besar bobot di perbaiki (untuk backpropagation)

def praproses(path):                        # resize dulu, baru grayscale, lalu skala 0..1
    img = Image.open(path).convert("RGB").resize((UKURAN, UKURAN), Image.LANCZOS).convert("L")
    return (np.asarray(img, dtype=np.float64) / 255.0).reshape(-1)     # vektor x (1024)

def muat(sub):                              # baca semua gambar train/valid -> X dan target T
    X, T = [], []
    for nama, pola in KELAS.items():
        d = os.path.join(FOLDER, sub, nama)
        for fn in (sorted(os.listdir(d)) if os.path.isdir(d) else []):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                X.append(praproses(os.path.join(d, fn))); T.append(pola)
    return np.array(X), np.array(T, dtype=np.float64)

def maju(x, v_0j, v_ij, w_0k, w_jk):        # PROPAGASI MAJU
    z_j = f(v_0j + x @ v_ij)                # imput -> hidden
    y_k = f(w_0k + z_j @ w_jk)              # hidden -> output
    return z_j, y_k

if __name__ == "__main__":                                                                      
    if not os.path.isdir(FOLDER): sys.exit(f"Folder '{FOLDER}' tidak ditemukan.")               #memastikan folder dataset ada
    X, T = muat("train"); Xv, Tv = muat("valid")                                                
    print(f"Arsitektur: 1 hidden layer | input X = {n} | hidden Z = {p} | output Y = {m}")
    print(f"Data latih: {len(X)} | validasi: {len(Xv)}")

    # Step 0: nilai awal pada bobot V dan W
    rng = np.random.default_rng(1)
    v_ij = rng.uniform(-0.5, 0.5, (n, p)) * (1.0 / np.sqrt(n)) * 5
    v_0j = rng.uniform(-0.5, 0.5, p)
    w_jk = rng.uniform(-0.5, 0.5, (p, m))
    w_0k = rng.uniform(-0.5, 0.5, m)

    for epoch in range(1, EPOCH_MAKS + 1):                 # Step 1: pengulangan epoch
        galat = 0.0
        for i in rng.permutation(len(X)):                  # Step 2: acak urutan data latih
            x, t = X[i], T[i]
            z_j, y_k = maju(x, v_0j, v_ij, w_0k, w_jk)     # Step 3-5: propagasi maju (input -> hidden -> output)

            # ################ BACK PROPAGATION (Step 6 - 8) ################
            delta_k    = (t - y_k) * df(y_k)               # Step 6: menghitung error output
            delta_in_j = w_jk @ delta_k                    # Step 7: error output dikirim mundur ke hidden
            delta_j    = delta_in_j * df(z_j)              # Step 7: menghitung error hidden
            w_jk += alpha * np.outer(z_j, delta_k); w_0k += alpha * delta_k   # Step 8: update W
            v_ij += alpha * np.outer(x, delta_j);   v_0j += alpha * delta_j   #         update V
            # ###############################################################

            galat += 0.5 * np.sum((t - y_k) ** 2) 
        mse = galat / len(X) 
        if epoch % 25 == 0 or epoch == 1: print(f"Epoch {epoch:4d} | error rata-rata = {mse:.6f}")
        if mse < TARGET:                                   # Step 9: kondisi berhenti karena error sudah cukup kecil
            print(f"Berhenti di epoch {epoch}, error = {mse:.6f}"); break 

    akurasi = lambda X, T: 100.0 * sum(np.argmax(maju(x, v_0j, v_ij, w_0k, w_jk)[1]) == np.argmax(t) 
                                       for x, t in zip(X, T)) / max(len(X), 1)
    print(f"Akurasi latih    : {akurasi(X, T):.2f}%")
    print(f"Akurasi validasi : {akurasi(Xv, Tv):.2f}%")

    np.savez("bobot_vw.npz", v_0j=v_0j, v_ij=v_ij, w_0k=w_0k, w_jk=w_jk,
             UKURAN=UKURAN, p=p, kelas=np.array(list(KELAS)))   # simpan V dan W (1 file)
    print("Bobot V dan W disimpan ke 'bobot_vw.npz'")