# 🚀 PyTorch_Bolum-6.1---Transformers_LLM

## 📌 Açıklama
Bu repo, PyTorch ile Transformer tabanlı LLM çalışmalarını adım adım öğretmek için hazırlanmıştır. Tokenizasyon, model tasarımı, eğitim döngüleri, loss & metric hesaplamaları ve attention mekanizmalarını içerir.

---

## 📂 Dosya Yapısı
```bash
/LLM
│
├─ 1_Verı_Hazırlama_-Tokenizasyon/
│ ├─ code/            # Tokenizasyon ve Dataset implementasyonu
│ └─ teorikler/       # Tokenizasyon teorik anlatımı
│
├─ 2_Model_Tasarımı-_LLM/
│ ├─ code/            # Encoder & Decoder blokları implementasyonu
│ └─ teorikler/       # LLM mimarisi ve teorik anlatım
│
├─ 3_Tokenizasyon_ve_Model/   # Tokenizasyon ve model entegrasyonu
│
├─ 4_Search-Inference/
│ ├─ code/            # Inference ve search örnekleri
│ └─ teorikler/       # Teorik anlatım
│
├─ 5_Train_Loop_Eval/
│ ├─ code/            # Eğitim döngüsü ve evaluation implementasyonu
│ └─ teorikler/       # Eğitim mantığı, backprop, loss, optimizer
│
├─ 6_Tokenizasyon-Model-TrainLoop/  # Tüm pipeline tek dosyada
│
├─ 7_Compile&Loss/    # Compile ve loss fonksiyonları
│
├─ 8_Metrics/         # Metric fonksiyonları
│
├─ 9_Baştan_Sona_LLM/ # Komple LLM uygulama örneği
│
├─ Attention_Modülleri/
│ ├─ Cross_Attention/
│ │ ├─ code/         # Cross-Attention implementasyonu
│ │ └─ teorikler/    # Cross-Attention teorik anlatımı
│ ├─ Dot_Product_Attention/
│ │ ├─ code/         # Dot-Product Attention implementasyonu
│ │ └─ teorikler/    # Dot-Product Attention teorik anlatımı
│ ├─ MultiHead_Attention/
│ │ ├─ code/         # Multi-Head Attention implementasyonu
│ │ └─ teorikler/    # Multi-Head Attention teorik anlatımı
│ └─ Self_Attention/
│   ├─ code/         # Self-Attention implementasyonu
│   └─ teorikler/    # Self-Attention teorik anlatımı
│
└─ Uygulama-_LLM/
  ├─ gui/                        # Kullanıcı arayüzü
  ├─ model/                      # Adım adım eğittiğimiz model ve tokenizer
  ├─ açıklamalar/                # Uygulama diyagramı ve açıklamalar
  ├─ masked_cross_entropy/       # Maskeli CrossEntropy Loss implementasyonu
  └─ calculate_llm_metrics/      # LLM metrik hesaplama (PPL, BLEU vb.)

```

---

## ⚙ Kullanılan Teknikler ve Modüller

- 🧠 Encoder-Decoder LLM: Girdi metnini gizli temsile dönüştürür ve çıktı metni üretir.  
- 🔤 Tokenizasyon: HuggingFace AutoTokenizer veya özel tokenizatör ile girdi/çıktı metinlerini tensörlere dönüştürür.  
- 🔄 Transformer Blokları:  
  - Self-Attention: Girdinin kendi içindeki bağıntıları öğrenir.  
  - Cross-Attention: Decoder, encoder çıktısına bakarak anlamlı üretim yapar.  
  - Multi-Head Attention: Paralel attention başlıkları ile bilgi kapsar.  
  - Dot-Product Attention: Attention skorlarını hızlı hesaplar.  
- ⚡ Feed-Forward & LayerNorm: Lineer dönüşümler ve normalizasyon sağlar.  
- 🔗 Residual Connection: Katmanlar arası bilgi kaybını önler.  
- 🛡 DropPath & Dropout: Overfitting’i engeller.  
- 🎯 SwiGLU & GELU Aktivasyonları: Non-linearity sağlar.  
- 📉 Masked Cross-Entropy Loss: Pad tokenları dikkate almaz, kaybı doğru hesaplar.  
- 🧰 Optimizer ve Scheduler: AdamW ve get_linear_schedule_with_warmup ile öğrenme oranı kontrolü.  
- 🎲 Top-K / Top-P Sampling: Sequence generation sırasında olası tokenları filtreler.  
- 📊 Metrics: Accuracy, Top-K accuracy, Perplexity, Sent-BLEU ve Corpus-BLEU.  
- 🛑 NaN / Inf Kontrolleri: Numerical stabilite sağlar.  
- 💾 EMA (Exponential Moving Average): Ağırlıkları stabilize eder.  
- 📦 Seq2Seq Dataset Yapısı: Encoder input ve decoder target’larını organize eder.  
- 🚧 Attention Masking: Padding ve future token’ları gizler.  
- 📝 Generation Pipeline: Greedy, Top-K ve Top-P yöntemleri ile güvenli metin üretimi.  

---

## 📊 Veri Seti

- Kaynak: [HuggingFace – Renicames/turkish-law-chatbot](https://huggingface.co/datasets/Renicames/turkish-law-chatbot)
- Kullanım Alanları: Hukuki metinlerin özetlenmesi, hukuki sorulara yanıt verilmesi, anayasal kavramların anlaşılması.
- Lisans: Apache 2.0
- Kaynaklar: Türkiye Cumhuriyeti Anayasası, Hukuki Metinler ve Açıklamalar, Avukatlara Sıkça Sorulan Sorular

---

# ⚠ Notlar:  
- Kaydedilen .pt modeli boyut olarak çok yüksek olduğundan repoya dahil edilmemiştir.  
- Yapılan uygulamada **epoch sayısı 5 olarak ayarlanmıştır. Uygulamanın daha net bir sonuç vermesini istiyorsanız CUDA belleğine dikkat ederek epoch sayınını arttırabilirsiniz.**
- Tokenizer, modelin tüm adımlarında uyumlu şekilde kullanılmıştır.
