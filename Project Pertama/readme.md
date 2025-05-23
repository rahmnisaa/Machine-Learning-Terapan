# Laporan Proyek Machine Learning - Rahma Nur Annisa

## Domain Proyek

Polusi udara merupakan salah satu masalah lingkungan paling serius di dunia. Menurut laporan *Global Burden of Disease 2019*, polusi udara menjadi penyebab lebih dari **6,7 juta kematian** di seluruh dunia setiap tahunnya. Anak-anak sangat rentan terhadap dampaknya, dengan hampir **2.000 anak meninggal setiap hari** akibat paparan udara yang tercemar. (*World Health Organization, 2020*).

Polusi ini berkontribusi terhadap berbagai penyakit pernapasan seperti:

- **Penyakit Paru Obstruktif Kronik (PPOK)**
- **Pneumonia**
- **Kanker paru-paru**
- **Asma dan gangguan pernapasan lainnya**

Selain itu, paparan jangka panjang terhadap polutan seperti **PM2.5**, **PM10**, dan **NO₂** dapat menurunkan harapan hidup dan memperparah kondisi kesehatan masyarakat secara umum.

###  Mengapa Prediksi Kualitas Udara Penting?

Dengan prediksi kualitas udara yang akurat, kita dapat:

- Memberikan **peringatan dini** kepada masyarakat dan pemerintah.
- Menyusun kebijakan **pengurangan emisi** yang lebih tepat sasaran.
- Memprioritaskan **wilayah rawan** dengan kualitas udara buruk untuk tindakan lebih lanjut.
- Membantu **peneliti dan epidemiolog** dalam menghubungkan data polusi dengan tren kesehatan masyarakat.
---

Melalui proyek ini, diharapkan model machine learning dapat digunakan sebagai alat bantu prediktif untuk pengambilan keputusan dan mitigasi risiko kesehatan akibat polusi udara secara global.

## Business Understanding

### Problem Statements
1. Model machine learning apa yang paling tepat untuk memprediksi kualitas udara berdasarkan data polutan?  
2. Faktor apa saja yang paling berpengaruh dalam mempengaruhi prediksi kualitas udara?  
3. Bagaimana kontribusi masing-masing faktor terhadap hasil prediksi?

### Goals
1. Menentukan model klasifikasi terbaik untuk memprediksi kualitas udara.  
2. Mengidentifikasi variabel-variabel paling signifikan yang memengaruhi kualitas udara. 
3. Menjelaskan peran dan kontribusi setiap variabel dalam prediksi.

### Solution Statements  
Untuk mencapai tujuan tersebut, beberapa algoritma machine learning akan digunakan, seperti Decision Tree, Random Forest, Gradient Boosting, AdaBoost, CatBoost, XGBoost, dan Extra Trees. Model akan dievaluasi dengan metrik Balanced Accuracy agar hasilnya adil walaupun data tidak seimbang. Penanganan data imbalance dilakukan menggunakan metode SMOTETomek. Selanjutnya, hyperparameter tuning dilakukan pada model terbaik untuk meningkatkan performa.

## Data Understanding
Proyek ini menggunakan data dari **WHO Urban Ambient Air Pollution Dataset**, yang tersedia secara publik melalui tautan berikut:  
 [WHO Air Pollution Data](https://www.who.int/data/gho/data/themes/air-pollution)

Dataset mencakup data dari berbagai negara dan kota di dunia dengan fitur-fitur seperti:

- Konsentrasi polutan: `PM10`, `PM2.5`, dan `NO2`
- Informasi lokasi: `negara`, `kota`, `latitude`, `longitude`
- Karakteristik stasiun: `type_of_station`
- Label kualitas udara (`air_quality_category`) sesuai standar EAQI
- Fitur hasil rekayasa berbasis indeks kualitas udara: `pm10_tempcov`, `pm25_tempcov`, `no2_tempcov`

Dataset ini terdiri dari:
-  **25.000+ data latih**
-  **14.000+ data uji**

Distribusi geografis yang luas memungkinkan pemodelan prediktif yang **komprehensif dan global** terhadap kualitas udara.

###  Temuan Awal dari Data

- Wilayah **Asia Selatan dan Asia Tenggara** tercatat sebagai area dengan **kualitas udara terburuk**.
- Konsentrasi **PM10** dan **PM2.5** sangat berkorelasi tinggi.
- Wilayah dengan kualitas udara **"Dangerous"** memiliki nilai polutan yang jauh lebih tinggi dibandingkan kategori lainnya.
- Stasiun pengukuran di wilayah **urban** lebih sering melaporkan polusi tinggi.


## Data Preparation
Proses data preparation dilakukan secara bertahap dan terstruktur agar data siap digunakan untuk pemodelan machine learning dengan hasil yang optimal. Berikut adalah langkah-langkah detail yang dilakukan:

1. **Imputasi Missing Value**  
   Dataset memiliki beberapa nilai yang hilang (missing values) pada fitur numerik. Untuk menghindari kehilangan data yang penting dan agar model tetap dapat belajar dengan baik, nilai yang hilang diimputasi menggunakan **median** setiap fitur. Median dipilih karena lebih tahan terhadap outlier dibanding rata-rata.

2. **Encoding Data Kategorikal**  
   Data kategorikal seperti `country_name`, `city`, dan `type_of_station` perlu diubah menjadi format numerik agar bisa diproses oleh algoritma machine learning. Dua metode encoding digunakan:  
   - **One Hot Encoding**: Untuk fitur yang tidak memiliki urutan (nominal), seperti `country_name` dan `city`, agar setiap kategori diwakili oleh fitur biner.  
   - **Ordinal Encoding**: Untuk fitur yang memiliki urutan, misalnya `air_quality_category` yang memiliki kelas bertingkat seperti "Safety" dan "Dangerous".

3. **Feature Engineering berdasarkan European Air Quality Index (EAQI)**  
   Fitur baru dibuat berdasarkan rumus dan kategori EAQI untuk menangkap informasi tambahan terkait kondisi kualitas udara. Contohnya adalah `pm10_tempcov`, `pm25_tempcov`, dan `no2_tempcov` yang memberikan nilai pembobotan terhadap konsentrasi polutan sesuai standar EAQI.

4. **Penanganan Data Imbalance dengan SMOTETomek**  
   Dataset memiliki ketidakseimbangan kelas yang sangat besar, di mana kelas “Safety” jauh lebih banyak dibandingkan kelas “Dangerous”. Untuk mengatasi ini, digunakan teknik kombinasi **SMOTE (Synthetic Minority Over-sampling Technique)** dan **Tomek Links**:  
   - SMOTE berfungsi menambah data sintetis pada kelas minoritas agar seimbang dengan kelas mayoritas.  
   - Tomek Links menghilangkan data yang saling bertolak belakang dan menyebabkan overlap antara kelas sehingga data lebih bersih dan tegas.

5. **Preprocessing Terpisah antara Data Train dan Data Test**  
   Semua proses imputasi, encoding, feature engineering, dan penanganan imbalance hanya diterapkan pada **data train** menggunakan metode `.fit_transform()` agar model tidak "melihat" informasi dari data test. Pada data validasi dan data test, proses hanya dilakukan dengan `.transform()` menggunakan parameter yang sudah didapat dari data train. Ini penting untuk menghindari kebocoran data (data leakage) yang dapat membuat evaluasi model menjadi tidak valid.

Dengan rangkaian proses di atas, data yang awalnya mentah dan tidak lengkap menjadi bersih, lengkap, seimbang, dan siap digunakan untuk proses pemodelan machine learning.


## Modeling

Dalam proyek ini, berbagai model machine learning diuji untuk menentukan algoritma terbaik dalam memprediksi kualitas udara. Proses pelatihan menggunakan teknik resampling SMOTETomek untuk mengatasi ketidakseimbangan kelas pada data latih, kemudian model-model berikut dilatih dan dievaluasi:

1. **Decision Tree**  
   Model pohon keputusan yang sederhana dan mudah diinterpretasi. Meskipun rentan overfitting, model ini cepat dan efektif untuk baseline.

2. **Random Forest**  
   Ensemble dari banyak Decision Tree yang menggunakan teknik bootstrap dan pemilihan fitur acak. Model ini mengurangi risiko overfitting dan meningkatkan generalisasi.

3. **Gradient Boosting**  
   Metode boosting yang membangun model secara bertahap dengan fokus memperbaiki kesalahan model sebelumnya. Model ini efektif namun biasanya lebih lambat dibanding ensemble lain.

4. **AdaBoost**  
   Boosting yang menggunakan Decision Tree sebagai base learner dengan penyesuaian bobot pada data sulit. Cocok untuk data yang relatif bersih, tetapi kurang tahan terhadap noise.

5. **Extra Trees**  
   Mirip Random Forest, namun dengan split yang lebih acak, mempercepat pelatihan dan mengurangi varians model.

6. **XGBoost**  
   Gradient boosting yang dioptimasi untuk performa tinggi dengan regularisasi bawaan dan efisiensi komputasi. Sering menjadi pilihan utama dalam kompetisi machine learning.

7. **CatBoost**  
   Algoritma boosting yang unggul dalam menangani data kategorikal tanpa encoding eksplisit, serta otomatis mengatasi bias dan overfitting.

### Proses Pelatihan dan Evaluasi

- Data latih yang tidak seimbang diresampling menggunakan **SMOTETomek** untuk menggabungkan oversampling (SMOTE) dan undersampling (Tomek links), sehingga distribusi kelas menjadi lebih seimbang.  
- Model dilatih menggunakan data hasil resampling (`X_train_resampled`, `y_train_resampled`).  

## Evaluation
Pada proyek ini, metrik evaluasi yang digunakan adalah **Balanced Accuracy**. Balanced Accuracy dipilih karena dataset yang digunakan sangat tidak seimbang, dengan jumlah data kategori "Safety" jauh lebih banyak dibandingkan kategori "Dangerous". Metrik ini menghitung rata-rata sensitivitas (recall) dari setiap kelas, sehingga memberikan gambaran performa model yang lebih adil pada kelas minoritas.

Rumus Balanced Accuracy adalah:

$$
\text{Balanced Accuracy} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)
$$


di mana TP = True Positive, FN = False Negative, TN = True Negative, dan FP = False Positive.

Berikut hasil Balanced Accuracy dari model-model yang diuji:

| Model             | Balanced Accuracy |
|-------------------|-------------------|
| Decision Tree     | 0.9013            |
| Random Forest     | 0.9013            |
| Gradient Boosting | 0.9010            |
| AdaBoost          | 0.9013            |
| Extra Trees       | 0.9013            |
| CatBoost          | 0.9013            |
| **XGBoost**       | **0.9637**        |

Dari tabel di atas, dapat dilihat bahwa semua model selain XGBoost memiliki performa yang hampir sama dengan Balanced Accuracy sekitar 0.90. Namun, XGBoost menonjol dengan Balanced Accuracy sebesar 0.9637, menunjukkan kemampuan prediksi yang jauh lebih baik dan lebih seimbang antara kelas mayoritas dan minoritas.

Oleh karena itu, XGBoost dipilih sebagai model terbaik untuk tugas prediksi kualitas udara ini karena tidak hanya memiliki akurasi yang tinggi tetapi juga stabil dan mampu menangani imbalance data dengan baik.

### Post Analysis: Feature Importance

Setelah model XGBoost terpilih sebagai model terbaik, dilakukan analisis terhadap pentingnya fitur (feature importance) untuk memahami variabel mana yang paling berkontribusi terhadap prediksi kualitas udara.

Berikut adalah hasil feature importance dari model XGBoost:

- **who_region**: 0.5224  
  Fitur ini merupakan variabel dengan pengaruh terbesar, menunjukkan bahwa wilayah WHO sangat memengaruhi kualitas udara. Faktor geografis regional berperan penting dalam variasi kualitas udara.

- **pm10_concentration**: 0.1799  
  Konsentrasi PM10 merupakan polutan utama yang berkontribusi signifikan terhadap prediksi kualitas udara.

- **latitude**: 0.1370  
  Lokasi geografis (garis lintang) juga berpengaruh besar, kemungkinan terkait dengan kondisi iklim dan aktivitas lokal.

- **pm25_concentration**: 0.0147  
  Konsentrasi PM2.5 memiliki kontribusi kecil tetapi tetap relevan.

- **longitude**: 0.0144  
  Garis bujur memberikan kontribusi minor terhadap prediksi.

- **pm25_tempcov**: 0.0599  
  Fitur hasil rekayasa fitur yang berkaitan dengan variasi suhu dan PM2.5 juga berperan.

- **PM10_Category**: 0.0519  
  Kategori kualitas PM10 yang dipakai sebagai fitur tambahan memberikan kontribusi pada model.

- **who_ms**: 0.0119  
  Fitur WHO metadata tambahan memberikan pengaruh kecil.

- **no2_tempcov**: 0.0079  
  Variasi suhu terhadap NO2 memberikan kontribusi yang sangat kecil.

Fitur lainnya seperti konsentrasi NO2, jumlah stasiun pengukur, populasi, dan beberapa fitur lain memiliki kontribusi nol atau sangat kecil dalam model ini.

## Kesimpulan
![image](https://github.com/user-attachments/assets/ce2f0f34-c12d-4763-ba01-b14eb55a71cd)

Model XGBoost merupakan pilihan terbaik untuk memprediksi kualitas udara berdasarkan konsentrasi polutan. Faktor PM10 dan PM2.5 menjadi variabel yang paling berpengaruh dalam prediksi, sementara jenis stasiun pengukuran (urban atau rural) dan NO2 juga memberikan kontribusi. Model ini dapat dimanfaatkan untuk peringatan dini serta membantu pembuat kebijakan dalam mengendalikan kualitas udara.

## Rekomendasi

Untuk pengendalian kualitas udara, disarankan untuk fokus pada pengurangan konsentrasi PM10 dan PM2.5. Selain itu, pemantauan kualitas udara di wilayah urban harus diperkuat karena lebih berpengaruh terhadap kualitas udara. Model machine learning ini juga dapat dioptimalkan lebih lanjut untuk sistem deteksi dini guna mempercepat respons terhadap peningkatan polusi udara.

---
## Referensi
World Health Organization. (2020). Air pollution. Retrieved from https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health
