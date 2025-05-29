# LAPORAN MACHINE LEARNING (SISTEM REKOMENDASI) - NABIL DEFIN JATMIKO

## Project Overview

Industri permainan video (video game) telah bertransformasi menjadi salah satu sektor hiburan terbesar dan paling dinamis di dunia, dengan platform distribusi digital seperti Steam memainkan peran sentral. Steam sendiri menampung puluhan ribu judul game, menawarkan pilihan yang sangat beragam kepada jutaan penggunanya. Namun, volume konten yang masif ini seringkali menyebabkan pengguna mengalami kesulitan dalam menemukan game baru yang sesuai dengan selera dan preferensi unik mereka, sebuah fenomena yang dikenal sebagai _information overload_ atau kelebihan informasi. Proses pencarian manual yang mengandalkan penjelajahan etalase atau daftar populer bisa jadi tidak efisien dan seringkali membuat pemain melewatkan game-game yang sebenarnya akan mereka nikmati.

Pentingnya proyek ini terletak pada upaya untuk mengatasi tantangan penemuan game tersebut. Dengan menyediakan rekomendasi yang dipersonalisasi, pengalaman pengguna dapat ditingkatkan secara signifikan, membantu mereka menavigasi katalog game yang luas dengan lebih efektif dan menemukan judul-judul yang relevan. Pendekatan berbasis data dan Kecerdasan Buatan (AI), khususnya machine learning, menawarkan solusi yang objektif dan efisien untuk masalah ini, melampaui metode penemuan konvensional.

Proyek ini bertujuan untuk membangun sistem rekomendasi game menggunakan pendekatan content-based filtering yang ditingkatkan dengan deep learning. Dengan memanfaatkan dataset `steam.csv` yang berisi informasi detail mengenai berbagai game di platform Steam, proyek ini akan merekomendasikan game berdasarkan kemiripan atribut kontennya. Atribut-atribut utama yang digunakan meliputi genre game (`genres`), tag deskriptif dari SteamSpy (`steamspy_tags`), harga (`price`), serta skor rating yang dihitung dari kombinasi rating positif (`positive_ratings`) dan negatif (`negative_ratings`). Untuk menangani sifat multi-label dari `genres` dan `steamspy_tags`, teknik `MultiLabelBinarizer` akan diterapkan, sementara fitur numerik seperti `price` dan skor `ratings` akan dinormalisasi menggunakan `MinMaxScaler`.

Inti dari pendekatan ini adalah penggunaan model `Autoencoder` yang dibangun dengan `TensorFlow` dan `Keras` untuk mempelajari representasi fitur (embeddings) yang padat dan bermakna dari setiap game. Encoder dari autoencoder yang telah dilatih kemudian digunakan untuk menghasilkan vektor embedding untuk semua game dalam dataset. Kesamaan antar game dihitung menggunakan metrik `cosine similarity` pada embedding tersebut, yang menjadi dasar untuk menghasilkan rekomendasi. Sistem ini dirancang untuk dapat memberikan rekomendasi berdasarkan game input spesifik atau berdasarkan kombinasi filter seperti genre, rentang harga, dan rating minimum. Evaluasi model akan melibatkan analisis kurva `loss` pelatihan dan validasi dari autoencoder, serta pengukuran `precision@k` yang didasarkan pada kesamaan `steamspy_tags` antara game yang dijadikan acuan dengan game-game yang direkomendasikan. Proyek ini diharapkan tidak hanya menghasilkan sistem rekomendasi yang fungsional tetapi juga memberikan wawasan tentang efektivitas penggunaan autoencoder untuk feature learning dalam konteks rekomendasi game berbasis konten.

Penerapan machine learning, khususnya deep learning, dalam sistem rekomendasi berbasis konten telah menunjukkan potensi yang signifikan. Sebagai contoh, studi oleh Ferreira et al. (2020) mendemonstrasikan bagaimana autoencoder dapat digunakan untuk sistem rekomendasi produk, mengatasi masalah data sparsity dan menghasilkan rekomendasi yang relevan [1](https://www.mdpi.com/2076-3417/10/16/5510). Lebih lanjut, Bougteb et al. (2022) mengusulkan sistem rekomendasi hibrida berbasis deep autoencoder yang mampu mempelajari minat pengguna dan merekonstruksi rating yang hilang, menunjukkan performa yang lebih baik pada dataset berdimensi tinggi dibandingkan algoritma hibrida lainnya [2](https://www.igi-global.com/article/a-deep-autoencoder-based-hybrid-recommender-system/297963). Hasil-hasil ini mengindikasikan bahwa pendekatan yang diusulkan dalam proyek ini memiliki dasar yang kuat dan berpotensi memberikan kontribusi positif dalam membantu pemain menemukan game yang paling sesuai dengan preferensi mereka di platform Steam.

Referensi:
[1] Rekomendasi Sistem Menggunakan Autoencoder Oleh: Diana Ferreira, Sofia Silva, António Abelha, José Machado (2020) Tersedia di: MDPI
[2] Sistem Rekomendasi Hibrida Berbasis Deep Autoencoder Oleh: Yahya Bougteb, Bouchaib Ouhbi, Brahim Frikh, El Bachir Zemmouri (2022) Tersedia di: IGI Global

## Business Understanding

### Problem Statement

Dalam industri game yang terus berkembang pesat, menemukan game yang sesuai dengan preferensi pengguna merupakan tantangan signifikan. Dengan ribuan game yang dirilis setiap tahun, pengguna seringkali kewalahan dalam memilih. Sistem rekomendasi game yang ada saat ini seringkali memiliki keterbatasan dalam memahami preferensi pengguna secara mendalam, yang dapat menyebabkan:

- *Rekomendasi yang tidak relevan*: Pengguna menerima rekomendasi game yang tidak sesuai dengan genre, nama game, atau minimal rating dan harga yang mereka minati.
- *Kesulitan dalam menemukan game baru*: Pengguna mungkin melewatkan game-game menarik yang sebenarnya sesuai dengan selera mereka karena sistem rekomendasi yang kurang efektif dalam mengeksplorasi ragam pilihan.
- *Ketergantungan pada popularitas semata*: Banyak sistem hanya merekomendasikan game populer, mengabaikan game-game bagus yang mungkin kurang dikenal tetapi sangat cocok untuk pengguna.

### Goal

Proyek ini memiliki sasaran utama sebagai berikut:

- Mengembangkan model sistem rekomendasi game yang mampu merekomendasikan game yang relevan berdasarkan preferensi genre, nama permainan, atau harga dan rating pengguna.
- Meningkatkan kepuasan pengguna dengan menyediakan rekomendasi game yang lebih personal dan akurat.
- Membantu pengguna menemukan game baru yang sesuai dengan minat mereka, di luar daftar game populer.

### Solution Statement

Untuk mencapai tujuan di atas, proyek ini akan menerapkan solusi yang didasarkan pada machine learning dengan langkah-langkah yang dapat diukur:

- Pre-processing Data:
  - Mengimpor data game dari sumber yang relevan.
  - Melakukan pembersihan data, termasuk penanganan nilai yang hilang dan duplikasi.
  - Melakukan `MultiLabelBinarizer` pada kolom genre untuk mengonversi kategori genre menjadi format numerik yang dapat diproses oleh model.
  - Melakukan MinMaxScaler untuk menormalisasi fitur-fitur numerik, memastikan semua fitur memiliki skala yang serupa.
    
- Pengembangan Model `Autoencoder`:
  - Membuat arsitektur `Autoencoder` menggunakan `TensorFlow`/`Keras`. `Autoencoder` akan dilatih untuk mempelajari representasi laten dari genre game.
  - Input layer akan sesuai dengan jumlah genre unik.
  - Hidden layer akan mengompres informasi genre, menangkap pola-pola tersembunyi.
  - Output layer akan merekonstruksi kembali input genre.
  - Menggunakan `EarlyStopping` untuk mencegah overfitting selama pelatihan model, menghentikan pelatihan ketika performa pada data validasi tidak lagi meningkat.
    
- Sistem Rekomendasi Berbasis Kemiripan Kosinus:
  - Menggunakan `Cosine Similarity` antara representasi laten game untuk mengukur kemiripan antar game.
  - Menghitung skor kemiripan antara game yang dipilih pengguna dengan semua game lain dalam dataset.
  - Mengurutkan game berdasarkan skor kemiripan untuk menghasilkan daftar rekomendasi.
    
- Evaluasi Model:
  - Mengevaluasi kinerja sistem rekomendasi menggunakan metrik `Precision@K`. Metrik ini akan mengukur seberapa banyak rekomendasi teratas (K) yang relevan (misalnya, memiliki setidaknya satu genre yang sama dengan game target).
  - Melakukan evaluasi `rata-rata Precision@K` untuk sejumlah sampel game untuk mendapatkan gambaran menyeluruh tentang akurasi rekomendasi.

## Data understanding

Dataset yang digunakan dalam proyek ini adalah data game yang diperoleh dari Kaggle, dengan nama [Steam Games Dataset](https://www.kaggle.com/datasets/nikdavis/steam-store-games?select=steam.csv). Dataset ini berisikan informasi mengenai berbagai game yang tersedia di platform Steam, yang sangat relevan untuk membangun sistem rekomendasi game berdasarkan genre.

Dataset yang digunakan, yaitu steam.csv, terdiri dari 27.075 baris (game) dan 18 kolom (fitur/variabel). Setelah pemeriksaan awal, ditemukan bahwa dataset ini memiliki beberapa nilai yang hilang (missing values) pada beberapa kolom, seperti `developer` dan `publisher`. Berikut merupakan penjelasan fitur untuk seluruh fitur yang ada pada dataset:

- `appid`: ID unik untuk setiap game di Steam.
- `name`: Nama lengkap game.
- `release_date`: Tanggal rilis game.
- `english`: Indikator apakah game tersedia dalam bahasa Inggris (1: Ya, 0: Tidak).
- `developer`: Nama pengembang game.
- `publisher`: Nama penerbit game.
- `platforms`: Platform tempat game tersedia (windows, mac, linux).
- `required_age`: Batasan usia yang diperlukan untuk game.
- `categories`: Kategori fitur game (misalnya, Single-player, Multi-player, Co-op).
- `genres`: Genre game (misalnya, Action, Adventure, Indie).
- `steamspy_tags`: Tag SteamSpy yang menggambarkan genre dan karakteristik game. Fitur ini akan menjadi kunci dalam proyek ini.
- `achievements`: Jumlah achievements dalam game.
- `positive_ratings`: Jumlah ulasan positif dari pengguna.
- `negative_ratings`: Jumlah ulasan negatif dari pengguna.
- `average_playtime`: Waktu bermain rata-rata (dalam menit).
- `median_playtime`: Waktu bermain median (dalam menit).
- `owners`: Estimasi jumlah pemilik game (dalam rentang, misalnya 0-20000).
- `price`: Harga game dalam USD.

### Exploratory Data Analysis

Berikut adalah beberapa langkah eksplorasi data yang dilakukan untuk memahami karakteristik dataset:

- *Pengecekan Tipe Data*: Tipe data untuk setiap kolom diperiksa menggunakan `game_df.info()`. Ini membantu dalam mengenali variabel yang bersifat numerik dan kategorikal, serta mengidentifikasi kemungkinan masalah terkait tipe data yang tidak sesuai untuk analisis selanjutnya.
- *Statistika Deskriptif*: Statistik deskriptif seperti rata-rata, median, nilai minimum, maksimum, dan deviasi standar dihitung (`game_df.describe()`) untuk memberikan gambaran umum mengenai sebaran nilai pada variabel numerik. Ini memberikan wawasan awal tentang variasi keterampilan pemain dan distribusinya.
- *Menampilkan 5 data teratas*: Menampilkan 5 data teratas menggunakan `game_df.head()`. Ini membantu dalam visualisasi\gambaran isi dari dataset.
- *Pengecekan Missing Values*: Dilakukan perhitungan jumlah nilai yang hilang (`game_df. isnull(). sum()`). Sesuai instruksi, terungkap bahwa ada beberapa kolom yang memiliki nilai yang hilang, berikut fitur yang masih memiliki data NaN,
  ```
  developer	1
  publisher	14
  ```
- *Pengecekan Duplikasi Data*: Dilakukan pemeriksaan terhadap baris yang mungkin terduplikasi (`game_df[game_df. duplicated()]`). Dari hasil eksplorasi awal, tidak ditemukan baris yang terduplikasi, yang menunjukkan bahwa setiap entri adalah representasi pemain yang unik.
- *Visualisasi top 5 genre*: Visualisasi ini menampilkan lima genre game teratas berdasarkan jumlah game yang tersedia dalam dataset. Terlihat jelas bahwa "Indie" mendominasi dengan jumlah game sebanyak 19.421, jauh melampaui genre lainnya. Diikuti oleh "Action" dengan 11.903 game, "Casual" dengan 10.210 game, dan "Adventure" dengan 10.032 game. Genre "Strategy" berada di posisi kelima dengan 5.247 game. Data ini memberikan pemahaman awal tentang preferensi genre yang paling banyak direpresentasikan dalam kumpulan data ini.

## Data Preparation

### Menghapus fitur yang tidak digunakan 
