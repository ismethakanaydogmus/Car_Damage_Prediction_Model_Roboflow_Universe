# Car Damage Detection AI

Bu proje, fotoğraflardaki araba hasarlarını tespit eden, hasarın şiddetini hesaplayan ve arabanın hangi bölgesinde olduğunu sınıflandıran **YOLOv8m-seg (Instance Segmentation)** tabanlı bir yapay zeka uygulamasıdır. 

Roboflow Universe'deki geniş kapsamlı "Car Damage" kaza ve çarpışma veri setinde eğitilmiştir.

---

## 🌟 Özellikler

*   **Instance Segmentation**: Hasarlı bölgenin sadece etrafına kutu (bbox) çizmez, hasarın piksellerini (polygon) hassas olarak çevreler.
*   **Hasar Şiddeti ve Alanı**: Hasarın görselde kapsadığı yüzdeye göre hafif (🟢), orta (🟡), ağır (🟠), çok ağır (🔴) şeklinde şiddet sınıflandırması yapar.
*   **Bölge Tespiti**: Görseli 3x3 bir ızgaraya bölerek hasarın konumunu (örn: "Alt-Merkez", "Üst-Sağ") otomatik belirler.
*   **Masaüstü (CLI) Kullanımı**: Terminal üzerinden tek bir görseli analiz edip çıktıları (JSON raporu ve işaretlenmiş resim) alabilirsiniz.
*   **Gradio Web Arayüzü**: Son kullanıcılar için kullanımı oldukça basit, şık bir web paneli içerir.

## 📁 Proje Yapısı

*   `train.py`: RTX 3060 vs. cihazlar için optimize edilmiş, YOLOv8 eğitim (training) ve değerlendirme (evaluation) betiği. Kaldığı yerden devam etme (`--resume`) yeteneği vardır.
*   `predict.py`: Terminal (CLI) ortamından tekil görsellerde çıkarım (inference) yapmak, hasarlı görselleri renklendirmek ve JSON formatında hasar raporları çıkarmak için kullanılır.
*   `app.py`: Gradio 6.0 uyumlu, şık ve modern web arayüzünü başlatır. Sonuçları dinamik bir HTML tablosunda ve JSON sekmesinde listeler.
*   `requirements.txt`: Projenin bağımlılıkları.

## 🚀 Kurulum

Sisteminizde Python 3.9 veya daha üzeri bir sürümün yüklü olduğundan emin olun. 
Bağımlılıkları kurmak için terminalden proje dizinine gidip şu komutu çalıştırın:

```cmd
pip install -r requirements.txt
```

*(Not: Eğer CUDA destekli bir NVIDIA ekran kartınız varsa, bilgisayarınıza uygun PyTorch sürümünü [PyTorch sitesinden](https://pytorch.org/) indirerek nesne tespit sürelerini milisaniyelere indirebilirsiniz.)*

## 🕹️ Kullanım

### 1. Web Arayüzü İle (Önerilen)

En kolay yöntem web panelini başlatmaktır:

```cmd
python app.py
```

Tarayıcınızda [http://localhost:7860](http://localhost:7860) adresini açarak görsellerinizi sürükle-bırak yöntemiyle analiz edebilirsiniz. Kaydırıcıyı kullanarak model güvenilirlik (confidence) eşiğini ince veya kalın hasarlara göre ayarlayabilirsiniz.

### 2. Terminal (CLI) Üzerinden 

Tek bir görselin fotoğrafı üzerinden analiz yapmak, hasar raporunu `.json` ve işaretlenmiş çıktıyı `.jpg` olarak kaydetmek için:

```cmd
python predict.py --image "hasarli_araba.jpg" --conf 0.25
```

Çıktılar, ana dizindeki `predictions/` klasörü altına kaydedilir.

### 3. Modeli Tekrar Eğitmek (Training)

Kendi veri setinizle modeli sıfırdan veya var olan ağırlıkların üstüne eğitmek isterseniz:

```cmd
# Yeni eğitim başlatır (varsayılan: 50 epoch)
python train.py

# Daha uzun veya kısa eğitim
python train.py --epochs 100 --batch 8

# Yarıda kalan eğitime devam etmek için
python train.py --resume
```

## ⚠️ Dikkat Edilmesi Gerekenler

Bu model **kaporta ezikleri, kırıklar ve büyük gövde çizikleri** (standart kaza hasarları) üzerine eğitilmiştir. Boya atması, reçine, çok ince çizikler veya far/cam yansımaları tespit edilmeyebilir. Eğer bu tarz kılcal detayların da tespiti isteniyorsa, modelin bu tarz veri içeren fotoğraflarla yeniden eğitilmesi (fine-tuning) gerekmektedir. 

## 🛠️ Kullanılan Teknolojiler

*   **Ultralytics YOLOv8**: Model Mimarisi (segmentasyon)
*   **Gradio**: Web Kullanıcı Arayüzü
*   **OpenCV & NumPy**: Görsel İşleme ve Matematiksel Hesaplamalar
*   **PyTorch**: Derin Öğrenme Motoru

## Lisans

Bu proje kişisel / eğitim amaçlı bir çalışmadır. Model ağırlıkları *Roboflow Universe* kamu malı veri setleriyle üretilmiştir. İhtiyacınıza göre geliştirip kullanabilirsiniz.
