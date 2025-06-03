# Laporan Proyek Machine Learning - Rahma Nur Annisa

---

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dalam berbagai industri, termasuk hiburan digital seperti anime. Banyak pengguna menghadapi kesulitan dalam menemukan anime yang sesuai dengan selera mereka dari ribuan judul yang tersedia. Oleh karena itu, diperlukan sistem yang mampu menyarankan anime berdasarkan preferensi pengguna sebelumnya.

Masalah ini penting untuk diselesaikan karena dapat meningkatkan pengalaman pengguna, keterlibatan (engagement), dan kepuasan dalam mengakses platform streaming atau katalog anime. Menurut Ricci et al. (2015), sistem rekomendasi mampu meningkatkan penjualan, retensi pengguna, serta meminimalisasi beban pencarian dalam platform digital.

Dalam proyek ini, dibangun sistem rekomendasi berbasis konten (content-based filtering) dan collaborative filtering menggunakan dataset dari Kaggle: Anime Recommendation Database, dengan pendekatan TF-IDF dan cosine similarity, serta matrix factorization (SVD).

---

## Business Understanding

### Problem Statements

* Bagaimana memberikan rekomendasi anime yang relevan berdasarkan genre dan deskripsi dari anime yang pernah ditonton pengguna?
* Bagaimana menangani ketidakseimbangan popularitas anime, di mana hanya sedikit judul yang sangat populer dan sebagian besar kurang dikenal?

### Goals

* Mengembangkan sistem rekomendasi yang mampu menyarankan top-N anime yang relevan bagi pengguna berdasarkan genre dan informasi konten lainnya.
* Menyediakan hasil rekomendasi yang lebih inklusif dengan tidak hanya menampilkan anime populer, tetapi juga judul yang kurang dikenal namun relevan.

### Solution Approach

Untuk mencapai tujuan tersebut, digunakan dua pendekatan:

1. **Content-Based Filtering** menggunakan TF-IDF vectorizer pada fitur genre dan type, dan menghitung kemiripan antar anime dengan cosine similarity.
2. **Collaborative Filtering** menggunakan pendekatan matrix factorization (SVD) dengan library Surprise.

---

## Data Understanding

Dataset yang digunakan berasal dari Kaggle yang dapat diakses pada [https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) dan terdiri atas dua file:

* `anime.csv`: informasi anime seperti nama, genre, rating, episodes, dan member count (jumlah penonton).
* `rating.csv`: data interaksi pengguna terhadap anime (user\_id, anime\_id, rating).

### Ukuran Data:

* `anime.csv`: 12,294 entri
* `rating.csv`: 7,813,737 entri

### Fitur-fitur dalam `anime.csv`:

| Kolom     | Keterangan                                         |
| --------- | -------------------------------------------------- |
| anime\_id | ID unik untuk setiap anime                         |
| name      | Judul anime                                        |
| genre     | Daftar genre dalam bentuk string                   |
| type      | Tipe anime (TV, Movie, OVA, dll.)                  |
| episodes  | Jumlah episode                                     |
| rating    | Rata-rata rating dari pengguna                     |
| members   | Jumlah pengguna yang menandai anime di MyAnimeList |

* Terdapat missing values pada kolom **genre**, **type**, dan **rating**, serta outlier nilai rating **-1**.
* Genre paling umum adalah **Comedy**, **Action**, dan **Adventure**.
* Distribusi rating membentuk kurva normal dengan puncak di rating **6–7**.
* Distribusi members menunjukkan ketimpangan tinggi, hanya sebagian kecil anime yang sangat populer.

### Fitur-fitur dalam `rating.csv`:
| Nama Kolom | Deskripsi                                                                 |
|------------|---------------------------------------------------------------------------|
| user_id    | ID pengguna yang dihasilkan secara acak dan tidak dapat diidentifikasi.  |
| anime_id   | ID anime yang telah diberi rating oleh pengguna.                         |
| rating     | Nilai rating dari 1 hingga 10 yang diberikan pengguna.                   |
|            | Nilai -1 berarti pengguna telah menonton anime tersebut tetapi tidak memberi rating. |

---

## Data Preparation

Tahapan yang dilakukan dan alasannya:

1. **Handling Missing Values**
   Missing value pada kolom *genre* dan *type* dihapus karena kedua fitur ini merupakan komponen utama untuk membangun sistem rekomendasi berbasis konten. Jika dibiarkan, nilai kosong ini akan menghasilkan vektor kosong saat proses vektorisasi teks, yang berdampak negatif pada hasil perhitungan kemiripan antar anime.

2. **Removing Duplicates**
   Duplikat data dihapus untuk menghindari bias dalam model dan analisis, serta memastikan setiap anime hanya diwakili sekali dalam dataset. Duplikasi bisa menyebabkan hasil yang tidak akurat, baik dalam evaluasi maupun rekomendasi.

3. **Data Type Adjustment**
   Nilai "Unknown" pada kolom *episodes* diganti menjadi 0 dan diubah ke tipe data integer untuk memungkinkan analisis numerik. Nilai string pada kolom numerik akan menyebabkan error dalam pengolahan data lebih lanjut.

4. **Remove Invalid Ratings**
   Rating dengan nilai -1 menunjukkan bahwa pengguna belum memberikan rating yang valid. Data ini dihapus karena tidak dapat digunakan untuk melatih model collaborative filtering secara akurat.

5. **Fill Missing Ratings**
   Nilai kosong pada kolom rating diisi dengan rata-rata rating untuk menjaga integritas data numerik. Hal ini penting agar model tetap bisa dilatih secara stabil tanpa kehilangan banyak data.

6. **Combine Genre and Type**
   Genre dan type digabung menjadi satu kolom teks agar informasi konten lebih lengkap dan representatif ketika dilakukan vektorisasi. Ini memberikan konteks tambahan bagi TF-IDF dalam menangkap nuansa dari jenis dan isi anime.

7. **Text Vectorization**
   Kolom gabungan genre-type diubah ke dalam vektor numerik menggunakan TF-IDF. Proses ini mengubah informasi teks menjadi bentuk yang bisa dihitung kemiripannya menggunakan cosine similarity.

8. **Prepare Data for Surprise**
   Data rating disiapkan untuk modeling collaborative filtering dengan library Surprise. Dilakukan inisialisasi `Reader` untuk mendefinisikan skala rating.

9. **Data Splitting**
   Dilakukan pembagian data menjadi 80% data latih dan 20% data uji menggunakan `train_test_split`. Ini bertujuan untuk mengevaluasi performa model secara objektif pada data yang belum pernah dilihat sebelumnya.

```python
from surprise import Reader, Dataset, train_test_split

reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(rating_df[['user_id', 'anime_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
```

---

## Modeling

### 1. Content-Based Filtering

#### Algoritma

Model ini menggunakan TF-IDF untuk memproses teks genre dan type, lalu menghitung kemiripan antar anime menggunakan cosine similarity. Rekomendasi diberikan berdasarkan anime yang memiliki kemiripan konten tertinggi dengan judul yang sudah ditonton.

#### Kelebihan:

* Tidak memerlukan data pengguna secara eksplisit
* Cepat dan mudah diimplementasikan

#### Kekurangan:

* Tidak mampu memberikan rekomendasi yang berbeda secara konten (cold-start)
* Bergantung pada kualitas data konten

#### Contoh Kode:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_df['genre_type'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation Function
def get_recommendations(title):
    idx = anime_df[anime_df['name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return anime_df['name'].iloc[anime_indices]
```
Pada model yang dibangun pada project ini, dengan memberikan input judul anime Naruto, maka model akan memberikan 10 rekomendasi anime berikut

| Rank | Name | Genre | Type | Episodes | Rating | Members |
|------|------|-------|------|----------|--------|---------|
| 1 | Naruto: Shippuuden | Action, Comedy, Martial Arts, Shounen, Super Power | TV | 0 | 7.94 | 533578 |
| 2 | Naruto x UT | Action, Comedy, Martial Arts, Shounen, Super Power | OVA | 1 | 7.58 | 23465 |
| 3 | Rekka no Honoo | Action, Adventure, Martial Arts, Shounen, Super Power | TV | 42 | 7.44 | 35258 |
| 4 | Naruto Soyokazeden Movie | Action, Comedy, Martial Arts, Shounen, Super Power | Movie | 1 | 7.11 | 25174 |
| 5 | Boruto: Naruto the Movie | Action, Comedy, Martial Arts, Shounen, Super Power | Movie | 1 | 8.03 | 74690 |
| 6 | Naruto: Shippuuden Movie 4 | Action, Comedy, Martial Arts, Shounen, Super Power | Movie | 1 | 7.53 | 84527 |
| 7 | Naruto: Shippuuden Movie 3 | Action, Comedy, Martial Arts, Shounen, Super Power | Movie | 1 | 7.50 | 83515 |
| 8 | Project ARMS | Action, Martial Arts, Super Power | TV | 26 | 7.15 | 6903 |
| 9 | Kurokami The Animation | Action, Martial Arts, Super Power | TV | 23 | 7.29 | 72750 |
| 10 | Wolverine | Action, Martial Arts, Super Power | TV | 12 | 6.24 | 14989 |


### 2. Collaborative Filtering (Matrix Factorization - SVD)

#### Algoritma

Matrix factorization memecah matriks interaksi user-item menjadi dua matriks laten (user dan item). Digunakan pendekatan SVD (Singular Value Decomposition) yang populer dalam sistem rekomendasi seperti Netflix dan Amazon.

#### Kelebihan:

* Dapat menangkap hubungan kompleks antara pengguna dan item
* Memberikan rekomendasi lebih personal berdasarkan preferensi pengguna

#### Kekurangan:

* Memerlukan data pengguna dalam jumlah cukup
* Kurang efektif untuk user/item baru (cold-start)

#### Contoh Kode:

```python
from surprise import SVD
from surprise import accuracy

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)
```
Pada model yang dibangun pada project ini, user_id 1 diberikan rekomendasi 10 anime berikut


| Rank | Name                             | Genre                                                                 | Type  | Episodes | Rating | Predicted Rating | Members |
|------|----------------------------------|------------------------------------------------------------------------|--------|----------|--------|------------------|---------|
| 1    | Kimi no Na wa.                  | Drama, Romance, School, Supernatural                                 | Movie  | 1        | 9.37   | 10.00            | 200630  |
| 2    | Fairy Tail                      | Action, Adventure, Comedy, Fantasy, Magic, Shounen                   | TV     | 175      | 8.22   | 10.00            | 584590  |
| 3    | Fairy Tail (2014)               | Action, Adventure, Comedy, Fantasy, Magic, Shounen                   | TV     | 102      | 8.25   | 9.83             | 255076  |
| 4    | Tengen Toppa Gurren Lagann      | Action, Adventure, Comedy, Mecha, Sci-Fi                             | TV     | 27       | 8.78   | 9.77             | 562962  |
| 5    | Fullmetal Alchemist: Brotherhood| Action, Adventure, Drama, Fantasy, Magic, Military, Shounen          | TV     | 64       | 9.26   | 9.76             | 793665  |
| 6    | Major: World Series             | Comedy, Drama, Sports                                                | OVA    | 2        | 8.50   | 9.74             | 13405   |
| 7    | Katekyo Hitman Reborn!          | Action, Comedy, Shounen, Super Power                                 | TV     | 203      | 8.37   | 9.73             | 258103  |
| 8    | Gintama°                        | Action, Comedy, Historical, Parody, Samurai, Sci-Fi                  | TV     | 51       | 9.25   | 9.71             | 114262  |
| 9    | Dragon Ball Z                   | Action, Adventure, Comedy, Fantasy, Martial Arts, Shounen, Super Power | TV     | 291      | 8.32   | 9.69             | 375662  |
| 10   | Ginga Eiyuu Densetsu            | Drama, Military, Sci-Fi, Space                                       | OVA    | 110      | 9.11   | 9.68             | 80679   |




---

## Evaluation

### Content-Based Filtering

* Digunakan metrik **Precision\@K** dan **Recall\@K**.
* Precision\@10 dan Recall\@10 bernilai **1**, yang berarti semua rekomendasi yang diberikan relevan terhadap histori pengguna.

#### Formula:

- **Precision@K** = jumlah rekomendasi relevan ⁄ K  
- **Recall@K** = jumlah rekomendasi relevan ⁄ jumlah item relevan yang tersedia


### Collaborative Filtering

* Dievaluasi menggunakan **Root Mean Square Error (RMSE)** pada test set.
* RMSE mengukur rata-rata selisih kuadrat antara nilai rating aktual dan prediksi.

#### Formula:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(\hat{r}_i - r_i)^2}
$$

* Hasil evaluasi: **RMSE = 1.1320** (semakin rendah semakin baik)

---

## Referensi

* Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
* Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
* Kaggle. (2017). *Anime Recommendation Database*. Retrieved from: [https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
* Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). *Item-based collaborative filtering recommendation algorithms*. [WWW](http://WWW).
* Desrosiers, C., & Karypis, G. (2011). *A comprehensive survey of neighborhood-based recommendation methods*. In *Recommender Systems Handbook* (pp. 107-144).
