# Prediksi Risiko Gagal Bayar Home Credit

> Proyek Data Science End-to-End — Project Based Internship Home Credit  
> Fokus: Analisis Risiko Kredit, Pengolahan Data Relasional, Insight Bisnis, dan Machine Learning

## 1. Gambaran Umum Proyek

### Permasalahan Bisnis

Home Credit bertujuan memperluas akses layanan keuangan bagi pelanggan yang mungkin memiliki riwayat kredit terbatas atau tidak lengkap. Salah satu tantangan utama dalam pemberian pinjaman konsumen adalah menilai apakah pemohon berpotensi mengalami kesulitan pembayaran setelah menerima pinjaman.

Keputusan kredit yang didasarkan pada informasi tidak lengkap dapat menimbulkan dua risiko bisnis:

- **Risiko kredit:** pemohon yang tidak mampu membayar kembali pinjaman dapat disetujui, sehingga meningkatkan potensi gagal bayar dan biaya penagihan.
- **Hilangnya peluang bisnis:** pemohon yang sebenarnya mampu membayar dapat ditolak karena tingkat risikonya dinilai secara tidak akurat.

Proyek ini mengembangkan alur kerja machine learning untuk memperkirakan probabilitas terjadinya kesulitan pembayaran (`TARGET = 1`) pada setiap pengajuan pinjaman. Probabilitas yang dihasilkan dapat mendukung penilaian dan prioritas risiko. Namun, hasil tersebut tidak boleh diperlakukan sebagai keputusan pemberian pinjaman otomatis tanpa validasi lebih lanjut, aturan kebijakan, dan pengawasan manusia.

### Tujuan Proyek

1. Memahami profil pemohon dan perilaku kredit historis dari beberapa dataset yang saling berhubungan.
2. Membersihkan dan mengintegrasikan informasi pengajuan, biro kredit, pinjaman sebelumnya, cicilan, POS cash, dan kartu kredit.
3. Mengeksplorasi pola yang berkaitan dengan kesulitan pembayaran serta menerjemahkan temuan yang didukung data menjadi rekomendasi bisnis.
4. Membangun dan membandingkan model klasifikasi untuk memprediksi risiko gagal bayar.
5. Menghasilkan skor probabilitas dan segmentasi risiko untuk pengajuan yang belum memiliki label.

## 2. Pemahaman Dataset

### Sumber Dataset

Proyek ini menggunakan dataset **Home Credit Default Risk** dari kompetisi Kaggle:

[Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

Dataset terdiri dari satu tabel utama pengajuan pinjaman dan enam tabel historis terkait. Tabel utama memiliki satu baris untuk setiap pengajuan pinjaman saat ini. Sementara itu, tabel historis memiliki beberapa catatan untuk setiap pemohon sehingga perlu diagregasi ke tingkat pemohon sebelum digabungkan dengan tabel utama.

### Inventaris Dataset

Jumlah baris berikut merupakan jumlah yang diamati pada notebook proyek.

| Dataset | Jumlah Baris | Deskripsi |
|---|---:|---|
| `application_train.csv` | 307,511 | Pengajuan pinjaman saat ini yang memiliki label target |
| `application_test.csv` | 48,744 | Pengajuan tanpa label target yang digunakan untuk prediksi |
| `bureau.csv` | 1,716,428 | Riwayat kredit sebelumnya yang dilaporkan oleh lembaga keuangan lain |
| `bureau_balance.csv` | 27,299,925 | Riwayat status bulanan kredit pada biro kredit |
| `previous_application.csv` | 1,670,214 | Riwayat pengajuan pinjaman sebelumnya ke Home Credit |
| `POS_CASH_balance.csv` | 10,001,358 | Snapshot bulanan pinjaman POS dan cash sebelumnya |
| `installments_payments.csv` | 13,605,401 | Catatan pembayaran cicilan terjadwal dan pembayaran aktual |
| `credit_card_balance.csv` | 3,840,312 | Informasi saldo dan pembayaran kartu kredit setiap bulan |

### Variabel Target

Target yang digunakan adalah `TARGET`:

- `0`: pengajuan diberi label tidak mengalami kesulitan pembayaran.
- `1`: pengajuan diberi label mengalami kesulitan pembayaran.

Data training memiliki distribusi berikut:

| Target | Jumlah Pengajuan | Proporsi |
|---|---:|---:|
| 0 — Tidak mengalami kesulitan pembayaran | 282,686 | 91.93% |
| 1 — Mengalami kesulitan pembayaran | 24,825 | 8.07% |
| **Total** | **307,511** | **100%** |

Rasio kelas 0 terhadap kelas 1 adalah sekitar **11.39 : 1**. Ini menunjukkan ketidakseimbangan kelas yang cukup besar. Oleh karena itu, akurasi saja tidak cukup untuk menilai model: model yang sebagian besar memprediksi kelas 0 dapat memperoleh akurasi tinggi, tetapi tetap melewatkan banyak pemohon yang mengalami kesulitan pembayaran.

### Karakteristik dan Tantangan Data

Notebook mengidentifikasi beberapa karakteristik penting pada dataset:

- Tabel utama training pada awalnya memiliki 307,511 baris dan 122 kolom, sedangkan tabel test memiliki 48,744 baris dan 121 kolom.
- Beberapa atribut terkait properti memiliki tingkat nilai hilang yang tinggi. Sebagai contoh, `COMMONAREA_AVG` memiliki sekitar 69.87% nilai hilang, sementara beberapa atribut apartemen dan bangunan memiliki tingkat nilai hilang di atas 60%.
- Variabel keuangan seperti `AMT_INCOME_TOTAL`, `AMT_CREDIT`, dan `AMT_GOODS_PRICE` memiliki nilai ekstrem.
- Banyak variabel numerik memiliki distribusi yang sangat menceng (skewed). `AMT_INCOME_TOTAL`, misalnya, memiliki nilai skewness sekitar 391.56 pada EDA awal notebook.
- Variabel kategorikal memiliki jumlah kategori yang beragam. Sebagai contoh, `ORGANIZATION_TYPE` memiliki 58 kategori unik pada data training.
- Tabel historis memiliki tingkat granularitas dan ukuran yang berbeda. Tabel-tabel tersebut tidak dapat langsung digabungkan ke tabel pengajuan tanpa agregasi karena dapat menyebabkan jumlah baris pengajuan berlipat ganda.

## 3. Analisis Eksplorasi Data dan Insight Bisnis

### Aktivitas EDA

Notebook melakukan beberapa analisis berikut:

- Analisis distribusi target.
- Analisis nilai hilang.
- Statistik deskriptif numerik dan visualisasi distribusi.
- Pemeriksaan skewness.
- Analisis korelasi antara variabel numerik dan `TARGET`.
- Analisis frekuensi variabel kategorikal.
- Pemeriksaan outlier pada variabel keuangan utama.

### Temuan Utama

**1. Skor kredit eksternal menunjukkan korelasi negatif terkuat dengan target di antara variabel numerik yang diperiksa.**

Notebook melaporkan korelasi Pearson berikut terhadap `TARGET`:

| Fitur | Korelasi |
|---|---:|
| `EXT_SOURCE_3` | -0.1789 |
| `EXT_SOURCE_2` | -0.1605 |
| `EXT_SOURCE_1` | -0.1553 |

Arah negatif menunjukkan bahwa skor yang lebih tinggi dari sumber eksternal berkaitan dengan frekuensi kesulitan pembayaran yang lebih rendah pada dataset ini. Hubungan tersebut merupakan asosiasi, bukan bukti hubungan sebab-akibat.

**Interpretasi dan rekomendasi bisnis**

Skor kredit eksternal merupakan sinyal risiko yang berguna pada dataset ini. Home Credit dapat mengevaluasi sinyal tersebut sebagai bagian dari penilaian kemampuan dan kelayakan kredit yang lebih menyeluruh, bukan dengan hanya mengandalkan satu skor atau batas nilai yang kaku.

Tindakan yang direkomendasikan:

- Memasukkan informasi skor eksternal ke dalam alur penilaian risiko apabila data tersebut tersedia secara sah dan sesuai untuk digunakan.
- Menggabungkan skor eksternal dengan indikator kemampuan finansial, kewajiban yang sedang berjalan, dan perilaku pembayaran.
- Memantau tingkat persetujuan, gagal bayar, dan penolakan pada setiap kelompok skor untuk menilai apakah ambang keputusan sudah tepat.
- Memeriksa performa model dan hasil keputusan pada segmen pelanggan yang relevan untuk mengidentifikasi potensi ketidakadilan atau dampak yang tidak diinginkan.

**2. Model mengidentifikasi trade-off deteksi risiko yang cukup berarti: model mampu mendeteksi sebagian pemohon yang mengalami kesulitan pembayaran, tetapi juga menandai banyak pemohon yang sebenarnya tidak mengalami kesulitan pembayaran.**

Pada threshold yang dipilih sebesar **0.6741**, confusion matrix out-of-fold XGBoost adalah:

| Aktual / Prediksi | Prediksi 0 | Prediksi 1 |
|---|---:|---:|
| Aktual 0 | 254,240 | 28,446 |
| Aktual 1 | 14,009 | 10,816 |

Berdasarkan hasil tersebut:

- Model mengidentifikasi 10,816 dari 24,825 kasus positif (recall: 43.57%).
- Model melewatkan 14,009 kasus positif (false negative).
- Model menandai 28,446 kasus negatif sebagai positif (false positive).
- Dari seluruh kasus yang ditandai positif oleh model, 27.55% benar-benar positif (precision).

**Interpretasi dan rekomendasi bisnis**

Prediksi positif dari model sebaiknya diperlakukan sebagai sinyal untuk pemeriksaan lebih lanjut, bukan sebagai alasan pasti untuk menolak pengajuan. Pada threshold ini, jumlah false positive cukup besar. Menolak secara otomatis seluruh pemohon yang ditandai model dapat mengecualikan banyak pemohon yang sebenarnya tidak mengalami kesulitan pembayaran.

Tindakan yang direkomendasikan:

- Menggunakan skor model untuk memprioritaskan pengajuan yang memerlukan peninjauan tambahan, verifikasi, atau pemeriksaan kemampuan pembayaran.
- Membuat kelompok peninjauan untuk pengajuan dengan tingkat risiko tidak pasti atau meningkat, alih-alih mengambil seluruh keputusan melalui satu threshold biner.
- Memilih threshold operasional berdasarkan perbandingan biaya antara gagal mendeteksi risiko gagal bayar dan penolakan yang tidak diperlukan.
- Memantau false positive, false negative, tingkat persetujuan, dan hasil pembayaran pelanggan setelah implementasi.
- Mengevaluasi kembali threshold menggunakan validation set terpisah dan asumsi biaya bisnis yang telah disetujui sebelum digunakan secara operasional.

### Temuan EDA Tambahan

- Beberapa variabel terkait properti memiliki nilai hilang di atas 60%. Notebook menghapus 17 kolom yang tingkat nilai hilangnya melebihi 60%.
- EDA awal mengidentifikasi nilai ekstrem pada variabel pendapatan, jumlah kredit, dan harga barang. Tahap preprocessing final membatasi nilai fitur numerik pada persentil ke-1 dan ke-99.
- `DAYS_BIRTH` memiliki korelasi positif sebesar 0.0782 terhadap `TARGET`, sedangkan `REGION_RATING_CLIENT_W_CITY` memiliki korelasi positif sebesar 0.0609. Korelasi tersebut relatif kecil dan tidak boleh ditafsirkan sebagai hubungan sebab-akibat secara mandiri.
- Notebook merekayasa fitur riwayat pembayaran, seperti frekuensi keterlambatan pembayaran dan durasi keterlambatan. Fitur-fitur tersebut berpotensi menjadi indikator perilaku yang berguna, tetapi notebook saat ini belum menyajikan analisis per segmen yang membuktikan dampak bisnis masing-masing fitur secara individual.

## 4. Prapemrosesan Data dan Rekayasa Fitur

Alur kerja memproses data pengajuan utama dan enam dataset historis sebelum membangun tabel final untuk pemodelan.

### 4.1 Pembersihan Data Pengajuan

Untuk `application_train` dan `application_test`, notebook melakukan langkah berikut:

- Mengganti nilai khusus `DAYS_EMPLOYED` yaitu `365243` dengan nilai hilang.
- Menghapus 17 kolom dengan lebih dari 60% nilai hilang, berdasarkan analisis missingness pada data training.
- Mengisi nilai hilang numerik menggunakan median dan nilai hilang kategorikal menggunakan modus.
- Membuat fitur terkait keuangan, demografi, pekerjaan, dan skor eksternal.

Contoh fitur hasil rekayasa pada data pengajuan:

| Fitur | Definisi / Tujuan |
|---|---|
| `CREDIT_INCOME_RATIO` | Jumlah kredit dibagi total pendapatan |
| `ANNUITY_INCOME_RATIO` | Anuitas dibagi total pendapatan |
| `CREDIT_ANNUITY_RATIO` | Jumlah kredit dibagi anuitas |
| `GOODS_CREDIT_RATIO` | Harga barang dibagi jumlah kredit |
| `INCOME_PER_FAMILY` | Pendapatan dibagi jumlah anggota keluarga |
| `AGE` | Usia dalam tahun yang diturunkan dari `DAYS_BIRTH` |
| `EMPLOYMENT_YEARS` | Durasi bekerja dalam tahun |
| `EXT_SOURCE_MEAN` | Rata-rata dari tiga skor sumber eksternal |
| `EXT_SOURCE_STD` | Standar deviasi skor sumber eksternal |
| `EXT_SOURCE_MIN` / `MAX` | Nilai minimum dan maksimum skor sumber eksternal |
| `EXT_SOURCE_PROD` | Perkalian dari tiga skor sumber eksternal |
| `EXT_SOURCE_WEIGHTED` | Skor berbobot: 0.2 × sumber 1 + 0.5 × sumber 2 + 0.3 × sumber 3 |
| `LOG_INCOME` / `LOG_CREDIT` / `LOG_ANNUITY` | Variabel keuangan yang ditransformasikan menggunakan log |
| `CREDIT_REQ_TOTAL` | Gabungan jumlah permintaan kredit harian, mingguan, dan bulanan dari biro kredit |

Fungsi pembagian yang digunakan menangani penyebut bernilai nol dan penyebut yang hilang dengan mengembalikan nilai nol.

### 4.2 Pemrosesan Data Historis

Tabel historis dikonversi menjadi ringkasan tingkat pemohon menggunakan `SK_ID_CURR`.

| Dataset | Pemrosesan utama dan fitur hasil rekayasa |
|---|---|
| `bureau` + `bureau_balance` | Jumlah kredit, jumlah kredit aktif dan lunas, jumlah utang, rasio utang terhadap kredit, indikator tunggakan, durasi kredit, dan ringkasan status biro kredit bulanan |
| `POS_CASH_balance` | Statistik keterlambatan pembayaran, frekuensi pembayaran terlambat, jumlah keterlambatan, progres cicilan, dan jumlah catatan |
| `credit_card_balance` | Rata-rata dan maksimum saldo, utilisasi kredit, rasio pembayaran, frekuensi keterlambatan, frekuensi penggunaan melebihi limit, dan jumlah transaksi penarikan |
| `previous_application` | Jumlah pengajuan sebelumnya, tingkat persetujuan, tingkat penolakan, tingkat pembatalan, rasio pengajuan terhadap kredit, rasio uang muka, dan rasio terkait pembayaran |
| `installments_payments` | Keterlambatan pembayaran, indikator pembayaran terlambat, rasio pembayaran, indikator kekurangan dan kelebihan pembayaran, serta agregat pembayaran |

Tabel historis diagregasi sebelum digabungkan sehingga setiap tabel hasil memiliki satu baris untuk setiap pemohon (`SK_ID_CURR`). Notebook memeriksa duplikasi ID pemohon pada tabel hasil agregasi sebelum integrasi.

### 4.3 Integrasi Data

Seluruh tabel historis yang telah diagregasi digabungkan ke data pengajuan menggunakan `SK_ID_CURR` dengan metode left join.

Dataset hasil integrasi memiliki:

- Training: **307,511 baris × 191 kolom** (termasuk `TARGET` dan `SK_ID_CURR`).
- Test: **48,744 baris × 190 kolom** (termasuk `SK_ID_CURR`).

Setelah integrasi:

- Nilai hilang pada jumlah catatan historis dan nilai agregat numerik diisi dengan nol.
- Nilai hilang kategorikal yang tersisa diisi dengan `Unknown`.
- Kolom fitur training dan test diselaraskan.
- `SK_ID_CURR` disimpan secara terpisah untuk keluaran prediksi dan dihapus dari fitur model.

### 4.4 Pembersihan Final dan Encoding

Notebook kemudian melakukan langkah berikut:

- Membatasi nilai fitur numerik pada persentil ke-1 dan ke-99.
- Menggunakan one-hot encoding untuk kolom kategorikal dengan kurang dari 10 nilai unik.
- Menggunakan `LabelEncoder` untuk kolom kategorikal dengan 10 nilai unik atau lebih; kategori pada data test yang tidak ditemukan saat training dipetakan ke `-1`.
- Menyelaraskan kolom hasil encoding pada data test dengan kolom training.
- Menghasilkan 236 fitur kandidat sebelum seleksi fitur.

### 4.5 Seleksi Fitur

Model XGBoost awal dilatih untuk menghitung tingkat kepentingan fitur (feature importance). Fitur dengan nilai importance lebih besar dari `0.001` dipertahankan.

- Jumlah fitur awal: 236
- Jumlah fitur setelah seleksi: **167**

Fitur terpilih digunakan oleh XGBoost dan Logistic Regression.

## 5. Implementasi dan Evaluasi Machine Learning

### 5.1 Formulasi Masalah

Proyek ini merupakan permasalahan supervised binary classification.

- Input: informasi pemohon serta fitur agregat historis kredit dan pembayaran.
- Target: `TARGET`.
- Output: estimasi probabilitas terjadinya kesulitan pembayaran.

### 5.2 Strategi Mengatasi Ketidakseimbangan Kelas

Target training memiliki rasio kelas 0 terhadap kelas 1 sekitar 11.39:1.

Notebook menggunakan:

- XGBoost: `scale_pos_weight = 11.39`.
- Logistic Regression: `class_weight = 'balanced'`.

Rasio ketidakseimbangan dihitung berdasarkan jumlah masing-masing kelas pada target training.

### 5.3 Model 1 — XGBoost Classifier

XGBoost dipilih sebagai model final karena menghasilkan metrik klasifikasi out-of-fold yang lebih baik dibandingkan model pembanding Logistic Regression pada notebook ini.

Konfigurasi utama:

| Parameter | Nilai |
|---|---:|
| `n_estimators` | 3,000 |
| `learning_rate` | 0.01 |
| `max_depth` | 4 |
| `min_child_weight` | 10 |
| `gamma` | 1.0 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.6 |
| `colsample_bylevel` | 0.6 |
| `colsample_bynode` | 0.7 |
| `reg_alpha` | 2.0 |
| `reg_lambda` | 10.0 |
| `scale_pos_weight` | 11.39 |
| `eval_metric` | `auc` |
| `early_stopping_rounds` | 150 |
| `tree_method` | `hist` |
| `device` | `cuda` |
| `random_state` | 42 |

Model menggunakan regularisasi, subsampling baris, dan subsampling fitur untuk mengendalikan kompleksitas model. Notebook mengatur akselerasi CUDA, sehingga eksekusi memerlukan instalasi XGBoost yang kompatibel dan lingkungan dengan dukungan CUDA.

Best iteration pada kelima fold adalah 2,996; 2,999; 2,999; 2,999; dan 2,997. Nilai-nilai ini mendekati batas 3,000 putaran yang dikonfigurasi. Oleh karena itu, hasil saat ini tidak dapat disebut sebagai bukti bahwa early stopping secara signifikan mengurangi jumlah pohon.

### 5.4 Model 2 — Logistic Regression

Logistic Regression digunakan sebagai model pembanding.

Konfigurasi:

| Parameter | Nilai |
|---|---|
| `C` | 0.05 |
| `penalty` | `l2` |
| `solver` | `saga` |
| `max_iter` | 1,000 |
| `class_weight` | `balanced` |
| `random_state` | 42 |

Sebanyak 167 fitur terpilih distandardisasi menggunakan `StandardScaler` untuk Logistic Regression. Scaler di-fit pada matriks fitur training terpilih, kemudian diterapkan pada matriks fitur test terpilih.

### 5.5 Strategi Validasi

Kedua model dievaluasi menggunakan **Stratified K-Fold cross-validation sebanyak 5 fold** dengan pengacakan dan `random_state=42`.

Notebook menghasilkan probabilitas out-of-fold (OOF) dengan memprediksi setiap fold validasi menggunakan model yang dilatih pada fold lainnya. Notebook juga mencatat skor ROC-AUC training dan validasi pada setiap fold untuk memeriksa selisih performa training dan validasi.

### 5.6 Performa Model

Metrik berikut diambil dari evaluasi OOF pada notebook.

| Metrik | XGBoost | Logistic Regression |
|---|---:|---:|
| OOF ROC-AUC | **0.78569** | 0.76713 |
| Average Precision | **0.27918** | 0.24736 |
| Accuracy | **0.8619** | 0.8488 |
| Precision — kelas 1 | **0.2755** | 0.2483 |
| Recall — kelas 1 | **0.4357** | 0.4307 |
| F1-score — kelas 1 | **0.3375** | 0.3150 |
| Rata-rata ROC-AUC validasi | **0.78571** | 0.76715 |
| Rata-rata ROC-AUC training | 0.82514 | 0.76932 |
| Rata-rata selisih AUC training-validasi | 0.03943 | **0.00217** |
| Standar deviasi AUC validasi | 0.00353 | **0.00331** |

**Pemilihan model:** XGBoost menjadi model yang dipilih dalam notebook ini karena memiliki performa lebih baik pada delapan dari sepuluh metrik perbandingan, termasuk ROC-AUC, Average Precision, accuracy, precision, recall, dan F1-score untuk kelas positif.

Logistic Regression memiliki selisih training-validasi yang lebih kecil dan standar deviasi ROC-AUC validasi antar-fold yang sedikit lebih rendah. Hal ini menunjukkan perilaku yang lebih stabil dan kompleksitas yang lebih rendah dalam perbandingan khusus ini, tetapi metrik kelas positifnya lebih rendah.

### 5.7 Optimasi Threshold dan Confusion Matrix

Notebook mencari threshold dari 0.05 hingga 0.95 dan memilih threshold yang memaksimalkan F1-score kelas positif pada prediksi OOF.

Threshold XGBoost terpilih: **0.6741**

Pada threshold tersebut:

| Metrik | Hasil |
|---|---:|
| Accuracy | 86.19% |
| Precision default | 27.55% |
| Recall default | 43.57% |
| F1-score default | 0.3375 |
| True negative | 254,240 |
| False positive | 28,446 |
| False negative | 14,009 |
| True positive | 10,816 |

Threshold ini merupakan titik operasi model, bukan threshold kebijakan kredit yang berlaku universal. Kesesuaiannya bergantung pada biaya bisnis, toleransi risiko, dampak terhadap pelanggan, serta validasi menggunakan data berlabel yang belum pernah dilihat model.

### 5.8 Prediksi pada Dataset Test

Tahap prediksi final menerapkan probabilitas rata-rata dari model XGBoost antar-fold pada 48,744 pengajuan test yang tidak memiliki label.

Ringkasan prediksi notebook:

| Output | Jumlah | Proporsi |
|---|---:|---:|
| Prediksi tidak gagal bayar (`0`) | 43,305 | 88.84% |
| Prediksi gagal bayar (`1`) | 5,439 | 11.16% |
| **Total** | **48,744** | **100%** |

Notebook juga mengelompokkan probabilitas prediksi ke dalam beberapa kategori risiko operasional:

| Kategori Risiko | Aturan Probabilitas | Jumlah | Proporsi |
|---|---|---:|---:|
| Risiko rendah | `< 0.20` | 13,127 | 26.93% |
| Risiko menengah | `0.20` hingga `< 0.40` | 15,972 | 32.77% |
| Risiko tinggi | `0.40` hingga `< 0.6741` | 14,206 | 29.14% |
| Risiko sangat tinggi | `>= 0.6741` | 5,439 | 11.16% |

**Penting:** angka tersebut merupakan label prediksi dan segmentasi berbasis probabilitas untuk dataset test yang tidak memiliki label. Angka tersebut bukan hasil gagal bayar yang teramati. Akurasi model pada test set tidak dapat dihitung dari dataset ini karena label targetnya tidak tersedia.

## 6. Rekomendasi Bisnis

Model sebaiknya diperlakukan sebagai komponen pendukung pengambilan keputusan, bukan pengganti kebijakan kredit secara menyeluruh.

### Rekomendasi 1 — Gunakan Sinyal Kredit Eksternal sebagai Bagian dari Penilaian Risiko Menyeluruh

Ketiga variabel sumber eksternal memiliki korelasi negatif numerik terkuat terhadap `TARGET` pada analisis korelasi awal notebook.

Tindakan:

1. Meninjau risiko pemohon berdasarkan kelompok skor eksternal.
2. Menggabungkan skor eksternal dengan pendapatan, beban kredit, dan indikator riwayat pembayaran.
3. Mengevaluasi hasil persetujuan dan gagal bayar pada setiap kelompok skor.
4. Menghindari penggunaan satu skor eksternal sebagai satu-satunya dasar persetujuan atau penolakan.

**Nilai bisnis yang diharapkan:** penilaian pemohon yang lebih terinformasi dan pendekatan yang lebih terstruktur untuk mengidentifikasi risiko, dengan tetap memerlukan validasi serta kepatuhan terhadap ketentuan penggunaan data yang berlaku.

### Rekomendasi 2 — Terapkan Peninjauan Berbasis Risiko, Bukan Penolakan Otomatis

Pada threshold OOF yang dipilih, model memiliki recall 43.57% dan precision 27.55% untuk kasus kesulitan pembayaran. Model juga menghasilkan 28,446 false positive dan 14,009 false negative.

Tindakan:

1. Menggunakan skor risiko untuk memprioritaskan peninjauan manual atau verifikasi kemampuan pembayaran tambahan.
2. Menentukan alur peninjauan untuk kategori risiko rendah, menengah, tinggi, dan sangat tinggi.
3. Membandingkan beberapa threshold alternatif berdasarkan estimasi kerugian kredit, kapasitas peninjauan, dan biaya penolakan yang keliru terhadap pemohon yang sebenarnya mampu membayar.
4. Memantau false positive, false negative, tingkat persetujuan, dan performa pembayaran aktual setelah implementasi.

**Nilai bisnis yang diharapkan:** kapasitas peninjauan yang lebih terarah serta cara yang transparan untuk menyeimbangkan risiko gagal bayar dengan risiko menolak pemohon yang sebenarnya mampu membayar.

### Langkah Lanjutan yang Direkomendasikan Sebelum Produksi

- Membangun ulang preprocessing dan seleksi fitur di dalam setiap fold cross-validation untuk mencegah data leakage.
- Melakukan fitting nilai imputasi, pemetaan kategori, dan batas outlier hanya pada data training, kemudian menggunakan transformasi yang sama pada data validasi dan test.
- Mengganti ordinal label encoding untuk kategori ber-cardinality tinggi dengan strategi encoding kategorikal yang sesuai atau model yang dapat menangani fitur kategorikal secara tepat.
- Melakukan tuning threshold pada validation set terpisah dengan asumsi biaya bisnis yang eksplisit.
- Mengevaluasi kalibrasi agar probabilitas prediksi dapat ditafsirkan sebagai estimasi risiko secara lebih andal.
- Melakukan pemeriksaan performa dan fairness pada tingkat segmen.
- Melakukan validasi pada dataset berlabel dari periode waktu berikutnya atau dataset independen sebelum deployment.
- Memantau data drift, model drift, realisasi gagal bayar, dan hasil keputusan secara berkala.

## 7. Alur Kerja Proyek

Proyek ini mengikuti alur kerja Data Science yang diharapkan:

1. **Riset Permasalahan (Problem Research)** — mendefinisikan permasalahan risiko kredit, variabel target, dan konteks bisnis.
2. **Prapemrosesan Data (Data Pre-Processing)** — membersihkan, merekayasa fitur, mengagregasi, dan mengintegrasikan tujuh dataset terkait.
3. **Visualisasi Data dan Insight Bisnis (Data Visualization and Business Insight)** — mengeksplorasi ketidakseimbangan kelas, nilai hilang, distribusi, korelasi, dan sinyal risiko.
4. **Implementasi dan Evaluasi Machine Learning (Machine Learning Implementation and Evaluation)** — melatih dan membandingkan XGBoost serta Logistic Regression menggunakan stratified cross-validation dan metrik OOF.
5. **Rekomendasi Bisnis (Business Recommendation)** — menerjemahkan perilaku model menjadi rekomendasi peninjauan dan pemantauan berbasis risiko.

## 8. Tools dan Teknologi

- Python
- Jupyter Notebook / Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib

## 9. Struktur Repository

Struktur repository GitHub yang disarankan:

```text
home-credit-default-risk/
│
├── _pbi_HomeCredit.ipynb
├── README.md
└── .gitignore
```

File CSV dataset mentah Home Credit tidak disertakan dalam repository ini. Unduh dataset dari halaman kompetisi Kaggle dan sesuaikan nilai `base_path` pada notebook agar mengarah ke direktori dataset lokal.

Jangan mengunggah dataset mentah, file model hasil pelatihan, atau path Google Drive pribadi ke repository publik kecuali terdapat alasan yang jelas dan izin untuk membagikannya.

## 10. Cara Menjalankan Proyek

1. Clone repository ini.
2. Unduh dataset Home Credit Default Risk dari Kaggle.
3. Ekstrak file CSV ke direktori dataset lokal.
4. Buka `_pbi_HomeCredit.ipynb` menggunakan Jupyter Notebook atau Google Colab.
5. Perbarui nilai `base_path` pada notebook agar mengarah ke direktori yang berisi file CSV.
6. Instal pustaka Python yang diperlukan jika belum tersedia.
7. Jalankan sel notebook secara berurutan.

Contoh instalasi:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

**Catatan lingkungan:** notebook mengatur `device='cuda'` untuk XGBoost. Jika CUDA tidak tersedia, gunakan instalasi XGBoost yang kompatibel dan ubah pengaturan device menjadi `cpu`. Hal ini dapat memengaruhi waktu eksekusi.

## 11. Keterbatasan dan Catatan Evaluasi

Metrik yang dilaporkan perlu ditafsirkan dengan mempertimbangkan keterbatasan berikut:

1. **Data leakage pada seleksi fitur:** feature selector XGBoost awal di-fit menggunakan seluruh matriks training berlabel sebelum proses validasi 5-fold. Dengan demikian, fold validasi turut memengaruhi pemilihan fitur. Hal ini dapat membuat metrik OOF menjadi terlalu optimistis.
2. **Potensi optimisme pada pemilihan threshold:** threshold dipilih untuk memaksimalkan F1 pada prediksi OOF yang sama dengan prediksi yang kemudian digunakan untuk melaporkan precision, recall, F1, dan confusion matrix berbasis threshold. Oleh karena itu, hasil evaluasi yang bergantung pada threshold tersebut bukan evaluasi final yang independen.
3. **Konsistensi preprocessing:** notebook mengisi nilai hilang dan membatasi outlier secara terpisah pada data train dan test. Untuk alur kerja yang siap produksi, transformasi tersebut sebaiknya di-fit pada data training dan transformasi hasil fitting yang sama diterapkan pada data validasi serta test.
4. **Encoding kategorikal:** `LabelEncoder` digunakan untuk variabel kategorikal dengan setidaknya 10 nilai unik. Untuk kategori nominal, metode ini dapat memperkenalkan urutan artifisial yang kemudian digunakan oleh pemisahan pada model tree.
5. **Batas early stopping:** best iteration XGBoost mendekati batas maksimum 3,000 putaran. Tuning lanjutan perlu memeriksa apakah batas yang lebih tinggi meningkatkan performa validasi tanpa mengurangi kemampuan generalisasi.
6. **Data test tanpa label:** 48,744 prediksi test tidak memiliki label target pada dataset ini. Akurasi 86.19% dan metrik klasifikasi lainnya merupakan estimasi dari validasi training OOF, bukan metrik pada test set.
7. **Tidak ada interpretasi kausal:** korelasi dan prediksi model menggambarkan pola dalam dataset; keduanya tidak membuktikan bahwa karakteristik pemohon tertentu menyebabkan kesulitan pembayaran.
8. **Belum divalidasi untuk produksi:** kalibrasi, validasi temporal independen, fairness, biaya operasional, dan pemantauan setelah deployment masih diperlukan sebelum model digunakan dalam keputusan kredit nyata.

Keterbatasan tersebut tidak menghilangkan pekerjaan yang telah diselesaikan. Sebaliknya, hal tersebut menjelaskan langkah lanjutan yang diperlukan agar analisis lebih andal dan sesuai untuk mendukung pengambilan keputusan di dunia nyata.

## 12. Referensi

1. Kaggle. (2018). [Home Credit Default Risk — Kompetisi dan Dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data). Deskripsi dataset pengajuan, biro kredit, pengajuan sebelumnya, cicilan, POS cash, dan catatan kartu kredit.
2. XGBoost Developers. [Parameter XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html). Referensi parameter model dan pelatihan, termasuk konfigurasi device.
3. XGBoost Developers. [Pengantar Paket Python — Early Stopping](https://xgboost.readthedocs.io/en/stable/python/python_intro.html). Penjelasan mengenai early stopping dan best iteration.
4. Scikit-learn Developers. [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html). Penjelasan validasi silang terstratifikasi.
5. Scikit-learn Developers. [Average Precision Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Definisi Average Precision untuk evaluasi precision-recall.
6. Scikit-learn Developers. [Kesalahan Umum dan Praktik yang Direkomendasikan](https://scikit-learn.org/stable/common_pitfalls.html). Panduan mengenai preprocessing yang tidak konsisten dan data leakage.
7. Scikit-learn Developers. [Prapemrosesan Data](https://scikit-learn.org/stable/modules/preprocessing.html). Panduan melakukan fitting transformasi pada data training dan menerapkannya secara konsisten.

---

**Disclaimer:** Repository ini merupakan proyek edukasi / portfolio project-based internship yang menggunakan dataset publik Home Credit Default Risk. Proyek ini bukan sistem credit scoring produksi resmi Home Credit dan tidak boleh digunakan untuk mengambil keputusan pemberian pinjaman secara nyata.
