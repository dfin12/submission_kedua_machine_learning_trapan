# LAPORAN MACHINE LEARNING (SISTEM REKOMENDASI) - NABIL DEFIN JATMIKO

## Project Overview

Industri permainan video (video game) telah bertransformasi menjadi salah satu sektor hiburan terbesar dan paling dinamis di dunia, dengan platform distribusi digital seperti Steam memainkan peran sentral. Steam sendiri menampung puluhan ribu judul game, menawarkan pilihan yang sangat beragam kepada jutaan penggunanya. Namun, volume konten yang masif ini seringkali menyebabkan pengguna mengalami kesulitan dalam menemukan game baru yang sesuai dengan selera dan preferensi unik mereka, sebuah fenomena yang dikenal sebagai _information overload_ atau kelebihan informasi. Proses pencarian manual yang mengandalkan penjelajahan etalase atau daftar populer bisa jadi tidak efisien dan seringkali membuat pemain melewatkan game-game yang sebenarnya akan mereka nikmati.

Pentingnya proyek ini terletak pada upaya untuk mengatasi tantangan penemuan game tersebut. Dengan menyediakan rekomendasi yang dipersonalisasi, pengalaman pengguna dapat ditingkatkan secara signifikan, membantu mereka menavigasi katalog game yang luas dengan lebih efektif dan menemukan judul-judul yang relevan. Pendekatan berbasis data dan Kecerdasan Buatan (AI), khususnya machine learning, menawarkan solusi yang objektif dan efisien untuk masalah ini, melampaui metode penemuan konvensional.

Proyek ini bertujuan untuk membangun sistem rekomendasi game menggunakan pendekatan content-based filtering yang ditingkatkan dengan deep learning. Dengan memanfaatkan dataset `steam.csv` yang berisi informasi detail mengenai berbagai game di platform Steam, proyek ini akan merekomendasikan game berdasarkan kemiripan atribut kontennya. Atribut-atribut utama yang digunakan meliputi genre game (`genres`), tag deskriptif dari SteamSpy (`steamspy_tags`), harga (`price`), serta skor rating yang dihitung dari kombinasi rating positif (`positive_ratings`) dan negatif (`negative_ratings`). Untuk menangani sifat multi-label dari `genres` dan `steamspy_tags`, teknik `MultiLabelBinarizer` akan diterapkan, sementara fitur numerik seperti `price` dan skor `ratings` akan dinormalisasi menggunakan `MinMaxScaler`.

Inti dari pendekatan ini adalah penggunaan model `Autoencoder` yang dibangun dengan `TensorFlow` dan `Keras` untuk mempelajari representasi fitur (embeddings) yang padat dan bermakna dari setiap game. Encoder dari autoencoder yang telah dilatih kemudian digunakan untuk menghasilkan vektor embedding untuk semua game dalam dataset. Kesamaan antar game dihitung menggunakan metrik `cosine similarity` pada embedding tersebut, yang menjadi dasar untuk menghasilkan rekomendasi. Sistem ini dirancang untuk dapat memberikan rekomendasi berdasarkan game input spesifik atau berdasarkan kombinasi filter seperti genre, rentang harga, dan rating minimum. Evaluasi model akan melibatkan analisis kurva `loss` pelatihan dan validasi dari autoencoder, serta pengukuran `precision@k` yang didasarkan pada kesamaan `steamspy_tags` antara game yang dijadikan acuan dengan game-game yang direkomendasikan. Proyek ini diharapkan tidak hanya menghasilkan sistem rekomendasi yang fungsional tetapi juga memberikan wawasan tentang efektivitas penggunaan autoencoder untuk feature learning dalam konteks rekomendasi game berbasis konten.

Penerapan machine learning, khususnya deep learning, dalam sistem rekomendasi berbasis konten telah menunjukkan potensi yang signifikan. Sebagai contoh, studi oleh Ferreira et al. (2020) mendemonstrasikan bagaimana autoencoder dapat digunakan untuk sistem rekomendasi produk, mengatasi masalah data sparsity dan menghasilkan rekomendasi yang relevan [[1]](https://www.mdpi.com/2076-3417/10/16/5510). Lebih lanjut, Bougteb et al. (2022) mengusulkan sistem rekomendasi hibrida berbasis deep autoencoder yang mampu mempelajari minat pengguna dan merekonstruksi rating yang hilang, menunjukkan performa yang lebih baik pada dataset berdimensi tinggi dibandingkan algoritma hibrida lainnya [[2]](https://www.igi-global.com/article/a-deep-autoencoder-based-hybrid-recommender-system/297963). Hasil-hasil ini mengindikasikan bahwa pendekatan yang diusulkan dalam proyek ini memiliki dasar yang kuat dan berpotensi memberikan kontribusi positif dalam membantu pemain menemukan game yang paling sesuai dengan preferensi mereka di platform Steam.

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

### Pemilihan fitur dan Penanganan Missing Values

#### Pemilihan fitur
```
game_df_selected = game_df[['name', 'genres', 'steamspy_tags', 'price', 'positive_ratings', 'negative_ratings']].copy()
```

Langkah pertama yaitu  memilih subset kolom tertentu dari DataFrame `game_df` yang asli. Kolom yang dipilih adalah `name`, `genres`, `steamspy_tags`, `price`, `positive_ratings`, dan `negative_ratings`. Hasil pemilihan ini kemudian disimpan dalam DataFrame baru bernama `game_df_selected`. Penggunaan `.copy()` memastikan bahwa `game_df_selected` adalah salinan independen, sehingga perubahan pada `game_df_selected` tidak akan mempengaruhi game_df asli. Alasan diperlukannya :
- Mengurangi Kompleksitas Model: Dengan hanya memilih fitur yang relevan, kita dapat mengurangi dimensi data, yang pada gilirannya dapat mempercepat proses pelatihan model dan membuatnya lebih mudah diinterpretasikan.
- Menghilangkan Fitur Tidak Relevan/Redundan: Data mentah seringkali berisi kolom yang tidak relevan atau redundan untuk tujuan analisis kita. Menghapusnya dapat meningkatkan kualitas model.
- Mengurangi Noise: Fitur yang tidak relevan bisa menjadi "noise" yang dapat mengganggu kinerja model. Dengan menghapusnya, kita dapat membantu model fokus pada informasi yang paling penting.
- Menghemat Memori: Memuat dan memproses seluruh dataset dengan banyak kolom yang tidak diperlukan bisa memakan banyak memori. Pemilihan fitur membantu menghemat sumber daya komputasi.

#### Penanganan Missing Values
```
game_df_selected.dropna(inplace=True)
```

Setelah memilih kolom yang relevan, langkah selanjutnya adalah menangani nilai-nilai yang hilang (missing values) dalam `game_df_selected`. Metode `dropna(inplace=True)` digunakan untuk menghapus seluruh baris dari DataFrame `game_df_selected` jika ada setidaknya satu nilai kosong (NaN) di salah satu kolom yang dipilih. Parameter `inplace=True` berarti perubahan (penghapusan baris) akan langsung diterapkan pada DataFrame `game_df_selected` itu sendiri, tanpa perlu menetapkan hasilnya kembali ke variabel. Alasan diperlukannya:
- Mencegah Error Pemodelan: Sebagian besar algoritma machine learning tidak dapat memproses data yang mengandung nilai kosong dan akan menghasilkan error. Menghapus baris yang memiliki nilai kosong adalah cara cepat untuk mengatasi masalah ini.
- Memastikan Integritas Data: Meskipun penghapusan baris dapat mengurangi ukuran dataset, ini memastikan bahwa setiap entri data yang digunakan dalam analisis atau pemodelan memiliki semua informasi yang diperlukan untuk kolom-kolom yang telah dipilih.
- Menghindari Bias: Terkadang, nilai yang hilang dapat mengindikasikan pola tertentu. Jika tidak ditangani, hal ini bisa menyebabkan bias dalam model. Namun, dalam kasus ini, penghapusan baris adalah pendekatan yang lebih sederhana yang mengasumsikan bahwa kehilangan data adalah acak atau dapat diterima.

### Transformasi Data
```
for col in ['genres', 'steamspy_tags']:
    game_df_selected[col] = game_df_selected[col].apply(lambda x: x.split(';') if isinstance(x, str) else [])
```

Transformasi data pada kolom `genres` dan `steamspy_tags`. Untuk setiap kolom ini, kode mengonversi string yang berisi beberapa nilai yang dipisahkan oleh titik koma (`;`) menjadi sebuah list. Proses ini menggunakan `apply(lambda x: x.split(';') if isinstance(x, str) else [])`, memastikan bahwa jika nilai adalah string, ia akan dipecah, dan jika bukan string, ia akan diubah menjadi list kosong. Transformasi ini dilakukan untuk menormalisasi format data menjadi list, yang memudahkan analisis lebih lanjut seperti penghitungan frekuensi atau persiapan untuk encoding.

### Rekayasa fitur 
```
game_df_selected['ratings'] = game_df_selected['positive_ratings'] / (game_df_selected['positive_ratings'] + game_df_selected['negative_ratings'] + 1e-5)
game_df_selected.drop(['positive_ratings', 'negative_ratings'], axis=1, inplace=True)
```

Rekayasa fitur rating dan pemilihan fitur tambahan. Sebuah kolom baru bernama `ratings` dibuat dengan menghitung rasio positive ratings terhadap total ratings (positif + negatif), ditambahkan 1e-5 untuk menghindari pembagian dengan nol. Setelah kolom `ratings` dibuat, kolom asli `positive_ratings` dan `negative_ratings` dihapus dari `game_df_selected`. Proses ini bertujuan untuk menciptakan fitur yang lebih informatif dan relevan, serta mengurangi redundansi data.

### Encoding Data
```
mlb_genres = MultiLabelBinarizer()
mlb_tags = MultiLabelBinarizer()

genres_encoded = mlb_genres.fit_transform(game_df_selected['genres'])
tags_encoded = mlb_tags.fit_transform(game_df_selected['steamspy_tags'])
```

Encoding data kategori multi-label untuk kolom `genres` dan `steamspy_tags`. Teknik yang digunakan adalah `MultiLabelBinarizer`. Prosesnya melibatkan inisialisasi dua objek `MultiLabelBinarizer` terpisah, satu untuk `genres` (`mlb_genres`) dan satu untuk `steamspy_tags` (`mlb_tags`). Kemudian, metode `fit_transform()` diterapkan pada masing-masing kolom yang sudah berbentuk list (`game_df_selected['genres']` dan `game_df_selected['steamspy_tags']`) untuk mengubahnya menjadi format biner (0 atau 1). Hasilnya disimpan dalam `genres_encoded` dan `tags_encoded`. Teknik ini sangat penting untuk mengonversi data kategori multi-label menjadi representasi numerik yang dapat dipahami oleh algoritma machine learning, di mana setiap genre atau tag unik menjadi kolom biner tersendiri, sehingga memungkinkan model untuk memproses informasi ini secara efektif.
```
features = np.hstack((genres_encoded, tags_encoded))
```

Setelah encoding kategori multi-label, dilakukan penggabungan fitur (feature concatenation). Teknik yang digunakan adalah `np.hstack()`. Prosesnya melibatkan penggabungan array `genres_encoded` dan `tags_encoded` secara horizontal (berdampingan). Ini berarti semua kolom biner yang mewakili `genres` akan digabungkan dengan semua kolom biner yang mewakili `steamspy_tags` menjadi satu array fitur tunggal bernama `features`. Tahap ini krusial untuk menciptakan satu set fitur komprehensif yang siap digunakan sebagai input untuk model machine learning.

### Feature Scaling
```
numerics = game_df_selected[['price', 'ratings']].values
```

Dua fitur numerik dipilih dari dataset, yaitu `price` dan `ratings`. Keduanya dianggap penting karena dapat memengaruhi preferensi pengguna terhadap sebuah game.
```
scaler = MinMaxScaler()
numeric_scaled = scaler.fit_transform(numerics)
```

Dilakukan transformasi data menggunakan teknik `MinMaxScaling` dari `sklearn.preprocessing`. Teknik ini mengubah nilai dalam fitur `price` dan `ratings` ke dalam rentang 0 hingga 1. Alasan diperlukannya:
- Perbedaan Skala: Nilai price dan ratings memiliki skala yang sangat berbeda. Misalnya, price bisa berkisar dari 0 hingga ratusan dolar, sementara ratings umumnya bernilai 0–10.
- Meningkatkan Kinerja Model: Banyak algoritma machine learning, terutama yang berbasis jarak seperti Content-Based Filtering atau K-Means, sensitif terhadap perbedaan skala antar fitur.
- Konsistensi Interpretasi: Menyamakan skala antar fitur membuat kontribusi masing-masing fitur menjadi setara saat digunakan dalam perhitungan jarak atau kemiripan.

### Final Feature Matrix
```
X = np.hstack((features, numeric_scaled))
```

Setelah fitur kategorikal seperti genre dan tag diubah menjadi representasi numerik melalui proses encoding, hasilnya digabungkan ke dalam variabel features menggunakan `np.hstack`. Selanjutnya, dilakukan proses penggabungan antara `features` tersebut dengan fitur numerik yang telah dinormalisasi (`price` dan `ratings`) menggunakan kembali fungsi `np.hstack`. Proses ini menyatukan seluruh fitur yang relevan, baik yang berasal dari data kategorikal (seperti genre dan tag) maupun numerikal ke dalam satu array `X` yang menjadi representasi akhir dari setiap entri game. Penggabungan ini penting untuk memastikan bahwa seluruh jenis informasi yang tersedia dapat diproses bersama oleh algoritma pemodelan, sehingga model dapat mempertimbangkan semua aspek karakteristik game dalam proses pembelajaran atau rekomendasi.

## Modeling dan Result

Sistem rekomendasi yang dibangun menggunakan pendekatan Content-Based Filtering berbasis representasi vektor fitur game. Untuk mengatasi dimensi fitur yang tinggi akibat encoding genre dan tag, serta untuk mengekstraksi representasi fitur yang lebih padat, digunakan arsitektur Autoencoder. Autoencoder merupakan jaringan saraf yang dilatih untuk merekonstruksi input-nya sendiri, sehingga dapat digunakan untuk menemukan representasi fitur laten dalam bentuk dimensi yang lebih rendah.
```
input_dim = X.shape[1]
encoding_dim = 32
```

Dari code diatas, jumlah fitur input (`input_dim`) berdasarkan jumlah kolom dari matriks fitur `X`, yang mencakup fitur kategorikal dan numerikal. Selanjutnya, ditetapkan ukuran encoding layer (`encoding_dim`) sebesar 32, yang berarti setiap game akan direpresentasikan dalam bentuk vektor berdimensi 32 setelah melalui proses encoding. Representasi ini kemudian digunakan untuk menghitung kemiripan antar game guna menghasilkan rekomendasi.

- Kelebihan:
  - Personalized: Rekomendasi dihasilkan berdasarkan karakteristik konten game yang relevan, sehingga tetap dapat memberikan saran meskipun belum ada data interaksi pengguna (cold-start untuk user).
  - Scalable: Representasi fitur yang lebih padat dari autoencoder membuat perhitungan kemiripan lebih efisien.
  - Interpretable: Dapat dijelaskan berdasarkan fitur game yang digunakan (genre, tag, harga, rating).
    
- Kekurangan:
  - Cold-Start pada Item: Game yang tidak memiliki informasi konten yang cukup (genre/tag kosong) akan sulit direkomendasikan.
  - Overfitting: Jika tidak dikontrol dengan baik, autoencoder bisa terlalu menyesuaikan pada data pelatihan.
  - Keterbatasan dalam Variasi Rekomendasi: Hanya merekomendasikan item yang mirip dengan preferensi awal, sehingga eksplorasi terhadap konten baru terbatas.

 ```
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
embedding = Dense(encoding_dim, activation='relu', name='embedding')(encoded)
decoded = Dense(64, activation='relu')(embedding)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)
```

Bagian encoding terdiri dari dua layer dense berturut-turut dengan 128 dan 64 neuron yang menggunakan aktivasi `ReLU`. Kemudian, dilanjutkan dengan layer utama bernama `embedding`, yang memiliki dimensi 32 (`encoding_dim`) dan berfungsi sebagai representasi laten dari setiap game. Setelah itu, bagian decoding dibangun secara simetris dengan dua layer dense yang masing-masing memiliki 64 dan 128 neuron, juga menggunakan aktivasi `ReLU`. Terakhir, output layer berukuran sama dengan input menggunakan aktivasi sigmoid untuk merekonstruksi kembali input asli.

Representasi dari layer embedding inilah yang digunakan untuk menghitung kemiripan antar game, sehingga sistem dapat merekomendasikan game-game yang paling mirip dengan preferensi pengguna dalam bentuk top-N recommendation. Pendekatan ini memungkinkan sistem memahami struktur fitur kompleks dari game secara efisien dan menghasilkan rekomendasi yang relevan berdasarkan kemiripan konten.

```
autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder = Model(inputs=input_layer, outputs=embedding)
```

Setelah arsitektur autoencoder selesai dirancang, dua model dibentuk untuk tujuan yang berbeda. Model pertama, autoencoder, merupakan model lengkap yang menghubungkan `input_layer` hingga `output_layer`, dan digunakan dalam proses pelatihan untuk merekonstruksi kembali input data. Sedangkan model kedua, encoder, dibentuk khusus untuk mengekstraksi representasi laten dari data input, yaitu output dari layer embedding. Representasi inilah yang kemudian digunakan untuk menghitung kemiripan antar game, karena dianggap mampu menangkap fitur utama dari setiap game dalam bentuk vektor berdimensi rendah. Dengan memisahkan model encoder, proses pembuatan rekomendasi menjadi lebih efisien tanpa perlu melibatkan seluruh struktur autoencoder.

```
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X, X, epochs=20, batch_size=64, validation_split=0.2, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
```

Setelah model autoencoder selesai dibangun, proses pelatihan dilakukan dengan menggunakan fungsi loss Mean Squared Error (`MSE`) dan optimizer `Adam`, yang dikenal efisien dalam menangani model neural network. Model dilatih untuk merekonstruksi input data `X` selama maksimum 20 epoch, dengan ukuran batch 64 dan porsi data validasi sebesar 20% dari keseluruhan data. Untuk mencegah overfitting, digunakan callback `EarlyStopping` dengan patience=3, yang secara otomatis menghentikan pelatihan jika tidak terjadi peningkatan pada data validasi selama tiga epoch berturut-turut. Selain itu, parameter `restore_best_weights=True` memastikan bahwa model akan mengembalikan bobot terbaik yang diperoleh selama pelatihan.

Hasil pelatihan ditunjukkan melalui grafik Autoencoder Training Loss, yang memperlihatkan penurunan yang tajam pada loss selama epoch awal dan kemudian stabil pada level yang rendah. Baik training loss maupun validation loss menunjukkan tren penurunan yang konsisten dan tidak menunjukkan tanda-tanda overfitting, karena kedua kurva tetap saling mendekati hingga akhir pelatihan. Hal ini menunjukkan bahwa model mampu belajar merekonstruksi data dengan baik dan generalisasi terhadap data validasi berjalan secara optimal.

```
game_embeddings = encoder.predict(X)

def recommend(game_name=None, genre=None, price_range=None, min_rating=None, top_k=20):
    filtered_indices = game_df_selected.index.tolist()

    if genre:
        filtered_indices = game_df_selected[game_df_selected['genres'].apply(lambda g: genre in g)].index.tolist()

    if price_range:
        low, high = price_range
        filtered_indices = [i for i in filtered_indices if low <= game_df_selected.loc[i, 'price'] <= high]

    if min_rating:
        filtered_indices = [i for i in filtered_indices if game_df_selected.loc[i, 'ratings'] >= min_rating]

    if game_name:
        if game_name not in game_df_selected['name'].values:
            return f"Game '{game_name}' tidak ditemukan."
        idx = game_df_selected[game_df_selected['name'] == game_name].index[0]
        query_vec = game_embeddings[idx].reshape(1, -1)
    else:
        if not filtered_indices:
            return "Tidak ada game yang cocok dengan kriteria."
        query_vec = game_embeddings[filtered_indices].mean(axis=0).reshape(1, -1)

    sim_scores = cosine_similarity(query_vec, game_embeddings)[0]
    top_indices = np.argsort(sim_scores)[::-1]

    top_filtered = [i for i in top_indices if i in filtered_indices][:top_k]
    return game_df_selected.iloc[top_filtered][['name', 'genres', 'price', 'ratings']]
```

Setelah model autoencoder selesai dilatih dan vektor representasi laten (`game_embeddings`) untuk setiap game berhasil diperoleh dari model encoder, dibangun sebuah fungsi bernama recommend untuk menghasilkan top-N rekomendasi berdasarkan kemiripan konten. Fungsi ini fleksibel karena dapat menerima beberapa parameter input opsional, seperti `game_name`, `genre`, `price_range`, dan `min_rating`, sehingga pengguna bisa mendapatkan rekomendasi yang disesuaikan dengan preferensinya. Proses awal dalam fungsi ini adalah menyaring daftar game sesuai dengan kriteria yang diberikan. Jika parameter `game_name` tersedia, maka sistem akan mencari game tersebut dan menggunakan vektornya sebagai query untuk perhitungan kemiripan. Namun, jika `game_name` tidak disebutkan, maka vektor query dibentuk dari rata-rata vektor game yang lolos filter.

Untuk menghitung kemiripan antar game, digunakan `cosine similarity` antara vektor query dan seluruh vektor game yang telah diekstraksi sebelumnya. Hasilnya diurutkan berdasarkan skor kemiripan tertinggi. Sistem kemudian menampilkan top 20 rekomendasi game yang paling relevan berdasarkan kemiripan konten, lengkap dengan informasi seperti nama, genre, harga, dan rating. Pendekatan ini memungkinkan sistem memberikan rekomendasi yang relevan baik berdasarkan kesamaan terhadap game tertentu maupun berdasarkan filter preferensi pengguna secara umum.

```
recommend(game_name="Counter-Strike")
```

Sistem rekomendasi dijalankan dengan input berupa nama game "Counter-Strike" untuk mencari 20 game yang paling mirip berdasarkan representasi laten yang telah dipelajari oleh model autoencoder. Dalam hal ini, sistem menggunakan cosine similarity untuk mengukur kedekatan antara vektor representasi game "Counter-Strike" dan seluruh game lainnya. Hasil yang diperoleh merupakan daftar game dengan konten yang paling relevan, yang umumnya berasal dari genre Action, serta menampilkan game dengan gaya permainan serupa. Tabel berikut menyajikan top 20 rekomendasi beserta nama game, genre, harga, dan skor rating-nya:

```
recommend(genre="Action")
```

Sistem rekomendasi juga dapat digunakan untuk mencari daftar game berdasarkan preferensi genre tertentu tanpa menyebutkan nama game tertentu. Dalam contoh ini, parameter genre diatur ke nilai "Action", sehingga sistem akan memfilter seluruh data game dan hanya mempertimbangkan game yang memiliki genre "Action". Selanjutnya, sistem menghitung rata-rata dari representasi embedding game bergenre tersebut untuk menghasilkan query vektor, yang kemudian dibandingkan dengan semua game menggunakan cosine similarity. Hasilnya adalah 20 game teratas yang paling mirip secara semantik dengan genre "Action", sebagaimana ditampilkan dalam tabel berikut:

```
recommend(price_range=(0, 10), min_rating=0.9)
```

Sistem rekomendasi juga dapat dikustomisasi berdasarkan preferensi harga dan kualitas game. Dalam contoh ini, pengguna tidak menyebutkan nama game atau genre tertentu, tetapi memberikan batasan harga antara 0 hingga 10 dan rating minimal 0.9. Sistem pertama-tama memfilter seluruh game berdasarkan kriteria tersebut, lalu menghitung rata-rata dari representasi vektor embedding game hasil filter tersebut untuk membentuk query vektor. Selanjutnya, sistem menghitung kemiripan (cosine similarity) antara vektor query tersebut dengan seluruh embedding game, dan mengembalikan 20 rekomendasi teratas berdasarkan skor kemiripan. Hasil dari pemanggilan fungsi ini ditunjukkan pada tabel berikut:


## Evaluation 

Metrik evaluasi yang digunakan adalah Precision@K, yang bertujuan untuk mengukur kualitas hasil rekomendasi berdasarkan kemiripan antara game target dengan game-game yang direkomendasikan. Metrik ini dipilih karena sesuai dengan konteks sistem rekomendasi berbasis konten (Content-Based Filtering), di mana keberhasilan sistem diukur berdasarkan seberapa relevan item-item yang direkomendasikan dibandingkan dengan preferensi pengguna atau fitur dari item acuan.

### Definisi dan Formula Precision@K

Precision@K didefinisikan sebagai proporsi dari K item teratas yang direkomendasikan yang dianggap relevan terhadap item acuan.

Secara matematis:



Dalam konteks proyek ini:
- Satu game dijadikan acuan (query).
- Sistem menghitung kemiripan (cosine similarity) antara game tersebut dengan seluruh game lainnya.
- Diambil Top-K game paling mirip (tidak termasuk dirinya sendiri).
- Dihitung berapa banyak dari Top-K tersebut yang memiliki kesamaan pada steamspy_tags dengan game acuan.
- Rasio inilah yang menjadi nilai Precision@K.

### Cara Kerja Metrik 

Kode precision_at_k menghitung Precision@K sebagai berikut:
- Mengambil vektor embedding dari game acuan.
- Menghitung cosine similarity terhadap seluruh game.
- Memilih K game paling mirip (dengan indeks tertinggi pada similarity).
- Menghitung berapa banyak dari K rekomendasi tersebut yang memiliki tag (`steamspy_tags`) yang saling overlap dengan game acuan.
- Jika ada minimal satu tag yang sama, maka rekomendasi tersebut dianggap relevan.

Sebagai contoh, jika dari 20 rekomendasi teratas (K=20), ada 15 game yang memiliki setidaknya satu tag yang sama dengan game acuan, maka:


### Hasil Evaluasi dan Interpretasi

Precision@K menunjukkan seberapa banyak rekomendasi yang relevan, tanpa mempertimbangkan urutan. Nilai precision yang tinggi (mendekati 1) berarti sistem mampu merekomendasikan game-game yang memiliki karakteristik serupa dengan game acuan, sesuai dengan pendekatan content-based yang digunakan (berbasis fitur deskriptif seperti genre, tag, dan embedding autoencoder).

Dalam eksperimen awal, `Precision@20` untuk game seperti "Counter-Strike" menunjukkan hasil yang cukup tinggi, karena sebagian besar game yang direkomendasikan memiliki genre dan tag yang sangat mirip (misalnya: Action, Shooter, Multiplayer). Hal ini menunjukkan bahwa model encoder berhasil menangkap representasi laten game secara efektif, dan sistem mampu memberikan rekomendasi yang relevan.

```
def precision_at_k(game_index, top_k=20):
    query_vec = game_embeddings[game_index].reshape(1, -1)
    sim_scores = cosine_similarity(query_vec, game_embeddings)[0]
    top_indices = np.argsort(sim_scores)[::-1][1:top_k+1]

    target_tags = set(game_df_selected.iloc[game_index]['steamspy_tags'])
    hits = 0
    for idx in top_indices:
        recommended_tags = set(game_df_selected.iloc[idx]['steamspy_tags'])
        if len(target_tags & recommended_tags) > 0:
            hits += 1
    return hits / top_k
```

Untuk mengevaluasi performa sistem rekomendasi yang dibangun, digunakan fungsi `precision_at_k`, yang menghitung nilai `Precision@K` dari suatu game terhadap rekomendasi yang dihasilkan oleh model. Dalam hal ini, digunakan nilai K = 20 untuk mencerminkan rekomendasi 20 game teratas. Fungsi bekerja dengan mengambil vektor representasi (embedding) dari game target berdasarkan hasil pelatihan autoencoder, lalu menghitung cosine similarity antara vektor tersebut dan seluruh game lainnya dalam dataset. Setelah diperoleh skor kemiripan, dipilih 20 game teratas (kecuali dirinya sendiri) sebagai hasil rekomendasi. Relevansi setiap game direkomendasikan dievaluasi berdasarkan kemiripan tag (`steamspy_tags`) dengan game acuan. Jika setidaknya satu tag cocok, maka itu dihitung sebagai "hit". Precision@20 dihitung sebagai rasio jumlah hit terhadap total 20 rekomendasi, memberikan ukuran seberapa akurat sistem dalam merekomendasikan game yang kontennya relevan.

```
sample_indices = np.random.choice(game_df_selected.index, size=1000, replace=False)
avg_precision = np.mean([precision_at_k(i, top_k=20) for i in sample_indices])
print(f"Average Precision@20: {avg_precision:.4f}")
```



## Referensi
[1] Rekomendasi Sistem Menggunakan Autoencoder Oleh: Diana Ferreira, Sofia Silva, António Abelha, José Machado (2020) Tersedia di: [MDPI](https://www.mdpi.com/2076-3417/10/16/5510)
[2] Sistem Rekomendasi Hibrida Berbasis Deep Autoencoder Oleh: Yahya Bougteb, Bouchaib Ouhbi, Brahim Frikh, El Bachir Zemmouri (2022) Tersedia di: [IGI Global](https://www.igi-global.com/article/a-deep-autoencoder-based-hybrid-recommender-system/297963)
