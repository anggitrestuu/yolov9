# Dokumentasi Penelitian Road Damage Detection menggunakan YOLOv9

## Gambaran Umum

Penelitian ini mengembangkan sistem deteksi kerusakan jalan menggunakan model YOLOv9 yang telah dilatih khusus untuk mendeteksi berbagai jenis kerusakan jalan seperti retak longitudinal, retak transversal, retak buaya, dan lubang jalan.

## Alur Penelitian

### 1. Training Model YOLOv9

**Platform yang Digunakan:** vast.ai dengan GPU RTX 4090

**File Training:** `TRAIN_YOLOv9_RDD2022.ipynb`

Pada tahap ini, model YOLOv9 dilatih menggunakan dataset RDD2022 (Road Damage Dataset 2022) untuk mengenali 4 jenis kerusakan jalan:
- Longitudinal (Retak memanjang)
- Transverse (Retak melintang) 
- Aligator (Retak buaya)
- Pathole (Lubang jalan)

**Mengapa menggunakan vast.ai?**
- Menyediakan akses GPU RTX 4090 dengan harga terjangkau
- Cocok untuk training model deep learning yang membutuhkan komputasi tinggi
- Tidak perlu investasi hardware mahal

**Hasil Training:**
- File model terlatih: `best.pt`
- Model ini berisi semua parameter yang diperlukan untuk training dan inference

### 2. Penyimpanan Model Custom

**Lokasi:** Folder `rdd2022/`

File `best.pt` yang dihasilkan dari training disimpan di folder `rdd2022` sebagai model custom hasil pelatihan. Model ini sudah dioptimalkan untuk mendeteksi kerusakan jalan berdasarkan dataset yang digunakan.

### 3. Reparameterisasi Model (YOLOv9-C ke GELAN-C)

**Tujuan:** Mengoptimalkan model untuk inference dengan menghapus parameter yang hanya diperlukan saat training.

**Proses:**
- Mengkonversi model YOLOv9-C menjadi GELAN-C
- Menghilangkan parameter tambahan yang hanya digunakan untuk training
- Mempertahankan akurasi deteksi sambil meningkatkan kecepatan inference

**Keuntungan Reparameterisasi:**
- **Kecepatan lebih tinggi:** Parameter berkurang drastis sehingga inference lebih cepat
- **Ukuran file lebih kecil:** Model menjadi lebih ringan
- **Efisiensi memori:** Menggunakan RAM lebih sedikit
- **Akurasi tetap terjaga:** Performa deteksi tidak menurun

**File yang Digunakan:** `tools/reparameterization.ipynb`

### 4. Interface Aplikasi dengan Streamlit

**Command untuk menjalankan:** `streamlit run app.py`

**Fitur Aplikasi:**
- Upload gambar untuk deteksi kerusakan jalan
- Pengaturan confidence threshold dan IOU threshold
- Pilihan device (CPU/GPU)
- Visualisasi hasil deteksi dengan bounding box
- Informasi detail tentang jenis dan tingkat kepercayaan deteksi

**File Interface:**
- `app.py` - Interface utama untuk deteksi gambar tunggal
- `lit/app.py` - Interface dengan fitur batch processing
- `lit/app_image.py` - Interface khusus untuk deteksi gambar

## Struktur Folder Penelitian

```
├── rdd2022/                    # Model custom hasil training
│   └── best.pt                 # Model terlatih
├── tools/                      # Tools untuk reparameterisasi
│   └── reparameterization.ipynb
├── app.py                      # Interface utama
└── TRAIN_YOLOv9_RDD2022.ipynb  # Notebook training
```

## Cara Menjalankan Penelitian

### Langkah 1: Training (Opsional - jika ingin training ulang)
1. Buka `TRAIN_YOLOv9_RDD2022.ipynb` di vast.ai atau platform GPU lainnya
2. Jalankan semua cell untuk memulai training
3. Tunggu hingga training selesai dan dapatkan file `best.pt`

### Langkah 2: Reparameterisasi
1. Buka `tools/reparameterization.ipynb`
2. Sesuaikan path model input dan output
3. Jalankan proses konversi dari YOLOv9-C ke GELAN-C

### Langkah 3: Menjalankan Interface
```bash
streamlit run app.py
```

### Langkah 4: Menggunakan Aplikasi
1. Buka browser dan akses aplikasi Streamlit
2. Upload gambar jalan yang ingin dideteksi
3. Atur parameter confidence dan IOU sesuai kebutuhan
4. Lihat hasil deteksi dengan bounding box dan label

## Jenis Kerusakan yang Dapat Dideteksi

1. **Longitudinal** - Retak yang memanjang searah dengan jalan
2. **Transverse** - Retak yang melintang tegak lurus dengan jalan  
3. **Aligator** - Retak yang membentuk pola seperti kulit buaya
4. **Pathole** - Lubang pada permukaan jalan

## Keunggulan Sistem

- **Akurasi Tinggi:** Model dilatih khusus untuk dataset kerusakan jalan
- **Kecepatan Optimal:** Reparameterisasi meningkatkan kecepatan inference
- **Interface User-Friendly:** Streamlit menyediakan interface yang mudah digunakan
- **Fleksibel:** Dapat dijalankan di CPU maupun GPU
- **Real-time:** Deteksi dapat dilakukan secara langsung pada gambar yang diupload

## Teknologi yang Digunakan

- **YOLOv9:** Model deteksi objek state-of-the-art
- **PyTorch:** Framework deep learning
- **Streamlit:** Framework untuk membuat web interface
- **OpenCV:** Library untuk pemrosesan gambar
- **vast.ai:** Platform cloud computing untuk training

## Kesimpulan

Penelitian ini berhasil mengembangkan sistem deteksi kerusakan jalan yang akurat dan efisien menggunakan YOLOv9. Dengan proses reparameterisasi, sistem dapat berjalan dengan kecepatan tinggi tanpa mengorbankan akurasi deteksi. Interface Streamlit memudahkan pengguna untuk menggunakan sistem tanpa perlu pengetahuan teknis yang mendalam.