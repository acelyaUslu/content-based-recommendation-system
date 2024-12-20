
# İçerik Temelli Filtreleme - Content Based Filtering

# Ürün içeriklerinin benzerlikleri üzerinden tavsiyeler geliştirilir.

# 1. Metinleri Matematiksel Olarak Temsil Et( Metinleri Vektörleşme )
# 2. Benzerlikleri Hesapla


# Metni matematiksel olarak ölçülebilir forma getirmeliyiz.
# Count Vector ( Word Count)
# TF - IDF

# Filmler arasında Öklid uzaklıklarını hesaplarız
# Euclidean Distance- Öklid uzaklığı
# Öklid uzaklığı : iki vektörün birbirleri arasındaki uzaklığın karesinin toplamlarının karekökü

# Cosine similarity
# benzerlik , similarity hesabı da yapabiliriz.

# Count Vectör Yöntemi : Sayım Vektörü

# 1- Eşsiz Tüm Terimleri Sütunlara, Bütün Dökümanları Satırlara yerleştir.
# 2- Terimlerin Dökümanlarda Geçme Frekanslarını Hücrelere Yerleştir.


# TF- IDF YÖNTEMİ : METİN VEKTÖRLEŞTİRM YÖNTEMİ

# Kelimelerin hem kendi metinlerinde hemde bütün odaklandığımız verideki geçme frekansları üzerinden
# bir normalizayon işlemi yapar.
# Genel bir standartlaştırma işlemi yapar.

# 1- Count Vectorizer'ı hesapla.(Kelimelerin her bir dokumandaki frekansı)
# 2- TF - Term Frequency' yi Hesapla. (t teriminin ilgili dokimandaki frekansı / dokumandaki toplam terim sayısı)
# 3- IDF Inverse Document Frequency'i Hesapla.
# ----1+ loge((toplam doküman sayısı+1) / ( içinde t terimi olan döküman sayısı +1)
# 4- TF * IDF 'i Hesapla
# 5- L2 Normalizsyonu Yap.( Satırların kareleri toplamının karekökünü bul, ilgili satırdaki tüm hücreleri bulduğun değere böl.)

# Proje : İçerik Temelli Tavsiye Sistemi - Content Based Recommender System
# Online film izleme platformu, kullanıcılarına film önerilerinde bulunmak istemektedir.
# Kullanıcıların login oranı çok düşük olduğu için kullanıcı alışkanlıkları toplayamıyor.
# Bu sebeple iş birlikçi filtreleme yöntemleri ile ürün önerileri geliştirilemiyor.
# ** Fakat kullanıcıların tarayıcıdaki izlerinden hangi filmleri izlediklerini bilmektedir
# Bu bilgiye göre film önerilerinde bulununuz.

# veri setinde 45000 film ile ilgili bilgi var.
# uygulama kapsamında film açıklamalarını içeren "overview" değişkeni ile çalışılacaktır.

# Content Based Recommedation ( İçerik Temelli Tavsiye)
# Film Overview'larına Göre Tavsiye Geliştirme

# 1 - TF-IDF Matrisinin Oluşturulması
# 2 - Cosine Similarity Matrisinin Oluşturulması
# 3 - Benzerliklerine Göre Önerilerin Yapılması
# 4 - Çalışma Scriptinin Oluşturulması

# 1 - TF-IDF Matrisinin Oluşturulması

import pandas as pd

pd.set_option("display.max_columns", None) # BÜTÜN sütunları göster.
pd.set_option("display.width", 500) # yan yana 500 column getir.
pd.set_option("display.expand_frame_repr", False) # Hepsini tek satırda göster.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("dataset/the_movies_dataset/movies_metadata.csv", low_memory=False) # DtypeWarning kapamak için
df.head()
df.shape

df["overview"].head()

# stopwords : ingilizcede and , in gibi ifadeler .ölçüm değeri tanımayan ifadeleri siliyoruz.
tfidf = TfidfVectorizer(stop_words="english")

df[df["overview"].isnull()]
df["overview"] = df["overview"].fillna('')

tfidf_matrix = tfidf.fit_transform(df["overview"])
# fit_transform : buradaki bilgileri kullanarak fit et ve dönüştür.
# fit etmek : ilgili veri yapısı üzerinde işlemi yapar.
# transform da eski değerlerle değiştirir. kalıcı hale getirir.
tfidf_matrix.shape
# satırlarda filmler, sütunlarda kelimeler var.

df["title"].shape

tfidf.get_feature_names()
tfidf_matrix.toarray()

# 2 - Cosine Similarity Matrisinin Oluşturulması

# çalışmadı !!!!!
cosine_sim = cosine_similarity(tfidf_matrix)
# cosine similarity fonksiyonu bana benzerliğini hespalamak istediğin matrisi ver der
# tek argüman ya da iki argümana girebiliriz.

cosine_sim.shape
cosine_sim[1] # 1. indeksteki filmin diğer filmlerle benzerlik skoru


# 3 - Benzerliklerine Göre Önerilerin Yapılması

indices = pd.Series(df.index, index=df["title"])
# filmlerin ismi , filmlerin indisi

indices.index.value_counts()

# duplicated metodu kullanıyoruz.
# title larda duplice var mı ona bakıyoruz.
# biz en son filmi tutmak istiyoruz. keep = last

indices = indices[~indices.index.duplicated(keep="last")]
indices["Cinderella"]

movie_index = indices["Sherlock Holmes"]
cosine_sim[movie_index]
similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df["title"].iloc[movie_indices]

# Çalışmanın Scriptinin Hazırlanması -- Preperation of Working Script

def content_based_recommender(title,cosine_sim, dataframe):
    # indexleri oluşturma
    indices = pd.Series(dataframe.index, index=dataframe["title"])
    indices = indices[~indices.index.duplicated(keep="last")]
    # title'ın indexini yakalama
    movie_index = indices[title]
    # title'a göre benzerlik skorlarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index],columns=["score"])
    # kendisi hariç ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score",ascending=False)[1:11].index
    return dataframe["title"].iloc[movie_indices]



content_based_recommender("Sherlock Holmes", cosine_sim, df)
content_based_recommender("The Matrix", cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words="english")
    dataframe["overview"] = dataframe["overview"].fillna("")
    tfidf_matrix = tfidf.fit_transform(dataframe["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim



cosine_sim = calculate_cosine_sim(df)
content_based_recommender("The Dark Knight Rises", cosine_sim, df)

# filmlerin idleri var. yanında bunlara önerilecek film idleri yer alacak.
# sql sorgusu ile bir film izlendiğinde önerilecek film belli olacak.



















