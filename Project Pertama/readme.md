# 🚀 Air Quality Prediction using Machine Learning

## 🧠 Domain Proyek

Polusi udara telah menjadi penyebab utama kematian akibat penyakit pernapasan seperti PPOK, pneumonia, dan kanker paru. Berdasarkan *Global Burden of Disease 2019*, hampir 2.000 anak meninggal setiap hari karena kualitas udara buruk.

Proyek ini bertujuan untuk:
- Memprediksi kualitas udara berdasarkan data konsentrasi polutan menggunakan machine learning
- Mengidentifikasi variabel yang paling berpengaruh terhadap kualitas udara
- Memberikan rekomendasi kebijakan berbasis data

📚 **Referensi**:
- GBD 2019 Risk Factors Collaborators. (2020). *The Lancet*, 396(10258), 1223–1249.
- WHO. *Air pollution*. [Link](https://www.who.int/health-topics/air-pollution)

---

## 🎯 Business Understanding

### 📌 Problem Statements
1. Model apa yang terbaik untuk memprediksi kualitas udara berdasarkan data polutan?
2. Apa saja faktor paling signifikan yang memengaruhi prediksi kualitas udara?
3. Bagaimana kontribusi masing-masing faktor terhadap prediksi?

### 🎯 Goals
1. Menentukan model klasifikasi terbaik
2. Mengidentifikasi variabel signifikan
3. Menjelaskan kontribusi tiap variabel

### 🛠 Solution Statements
- Menggunakan beberapa algoritma: Decision Tree, Random Forest, Gradient Boosting, AdaBoost, CatBoost, XGBoost, dan Extra Trees
- Evaluasi menggunakan **Balanced Accuracy**
- Penanganan data imbalance dengan **SMOTETomek**
- Hyperparameter tuning untuk model terbaik

---

## 📊 Data Understanding

- 📁 Dataset: WHO Urban Ambient Air Pollution Dataset ([link](https://www.who.int/data/gho/data/themes/air-pollution))
- 🔢 Jumlah data latih: 25.999 baris
- 🔢 Jumlah data uji: 14.005 baris

### 📌 Fitur penting
- **Numerik**: pm10_concentration, pm25_concentration, no2_concentration, number_of_station
- **Kategorik**: country_name, city, type_of_station, air_quality_category (label)
- **Feature Engineering**: pm10_tempcov, pm25_tempcov, no2_tempcov (berdasarkan EAQI)

### 📈 Exploratory Data Analysis
- Wilayah Asia Selatan & Tenggara = kualitas udara paling buruk
- PM10 dan PM2.5 sangat berkorelasi (r = 0.89)
- Wilayah dengan kualitas “Dangerous” memiliki konsentrasi polutan jauh lebih tinggi

---

## 🧹 Data Preparation

Langkah-langkah:
1. **Imputasi** missing value dengan median
2. **Encoding** kategorikal: One Hot & ordinal encoding
3. **Feature Engineering** berdasarkan European Air Quality Index (EAQI)
4. **SMOTETomek** untuk menangani class imbalance
5. Preprocessing dilakukan hanya pada data train untuk menghindari data leakage

---

## 🧪 Modeling

Model yang diuji:
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- CatBoost
- XGBoost
- Extra Trees

Splitting data: 80% train, 20% test  
Preprocessing dilakukan dengan `.fit_transform()` untuk train dan `.transform()` untuk valid/test

---

## 📏 Evaluation

### 🎯 Metrik: **Balanced Accuracy**

Balanced Accuracy digunakan karena dataset imbalance:
- 25.921 Safety
- 78 Dangerous

| Model         | Balanced Accuracy (%) |
|---------------|------------------------|
| XGBoost       | **96.37**              |
| Model lain    | [Lihat notebook]       |

📌 **Model terbaik:** XGBoost  
📌 **Alasan:** Akurasi tinggi dan stabil

---

## 🧠 Feature Importance

- 🔥 PM10 dan PM2.5 adalah faktor paling signifikan
- 🌆 Jenis stasiun (urban > rural) juga berkontribusi
- NO2 berpengaruh namun lebih kecil dibanding dua lainnya

---

## 📌 Conclusion

- XGBoost adalah model terbaik dalam memprediksi kualitas udara
- PM10 dan PM2.5 adalah faktor paling berpengaruh
- Model dapat digunakan untuk peringatan dini dan kebijakan publik

---

## ✅ Rekomendasi

- 🎯 Fokus pengendalian pada **PM10 dan PM2.5**
- 📍 Perkuat pemantauan di wilayah **urban**
- 📊 Optimalkan model ML untuk sistem **deteksi dini**

---

## 📁 Struktur Repo

