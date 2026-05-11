# 🛡️ AI-Powered Cybersecurity Assistant

> **Information Security Course Project**  
> Phishing Detection • Password Analysis • AI Security Chatbot

---

## 📌 Overview

A unified AI-powered cybersecurity platform that combines three modules to detect phishing URLs, analyze password strength, and provide real-time security guidance through an intelligent chatbot.

| Module | Model | Accuracy |
|--------|-------|----------|
| 🔗 Phishing URL Detection | Random Forest (200 trees) | **96.72%** |
| 🔐 Password Strength Analyzer | MLP Neural Network | **99.87%** |
| 🤖 AI Security Chatbot | LLaMA 3.3 70B via Groq | Real-time |

---

## 🗂️ Project Structure

```
AI-Cybersecurity-Assistant/
│
├── cybersecurity_notebook.ipynb   # Main Colab notebook (all 3 modules)
├── cybersecurity_app.py           # Streamlit web UI
├── dataset_phishing.csv           # Phishing URL dataset (11,430 URLs)
├── data.csv                       # Password strength dataset (15,000)

```

---

## 🔧 Modules

### 🔗 Module 1 — Phishing URL Detection
- **Algorithm:** Random Forest Classifier
- **Dataset:** 11,430 URLs (Kaggle — Hannousse & Yahiouche, 2021)
- **Features:** 87 URL-based features (length, special chars, HTTPS, subdomains, TLD, etc.)
- **Accuracy:** 96.72% | CV: 96.85% | ROC-AUC: >0.98

### 🔐 Module 2 — Password Strength Analyzer
- **Algorithm:** MLP Neural Network (128 → 64 → 32)
- **Dataset:** 15,000 passwords (Kaggle — Password Strength Classifier)
- **Features:** 22 engineered features (entropy, diversity, sequential patterns, etc.)
- **Classes:** Weak / Medium / Strong
- **Accuracy:** 99.87% | CV: 99.87% ± 0.04%

### 🤖 Module 3 — AI Security Chatbot
- **Model:** LLaMA 3.3 70B via Groq API
- **Type:** Multi-turn conversation with context memory
- **Topics:** Phishing, malware, passwords, network security, privacy, 2FA, VPNs
- **Response Time:** < 2 seconds

---

## 🚀 How to Run

### Option A — Google Colab
1. Open `cybersecurity_notebook.ipynb` in Google Colab
2. Upload `dataset_phishing.csv` and `data.csv` to `/content/`
3. Add your Groq API key in Module 3 cell
4. Run All cells

### Option B — Streamlit App (Local)
```bash
# Install dependencies
pip install streamlit scikit-learn pandas numpy groq tldextract

# Run the app
streamlit run cybersecurity_app.py
```
Then open `http://localhost:8501` in your browser.

### Option C — Streamlit on Google Colab
```python
!pip install streamlit pyngrok groq tldextract -q
from pyngrok import ngrok
import subprocess, threading, time

ngrok.set_auth_token("YOUR_NGROK_TOKEN")

def run():
    subprocess.run(["streamlit", "run", "cybersecurity_app.py",
                    "--server.port", "8501", "--server.headless", "true"])

threading.Thread(target=run, daemon=True).start()
time.sleep(10)
print(ngrok.connect(8501))
```

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
groq
tldextract
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📊 Datasets

| Dataset | Source | Size |
|---------|--------|------|
| Phishing URL Detection | [Kaggle — Hannousse & Yahiouche](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset) | 11,430 URLs |
| Password Strength | [Kaggle — Bhavikbb](https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset) | 670,000+ passwords |

---

## 🖥️ Streamlit UI Preview

The app includes 4 tabs:
- **🔗 Phishing Detector** — Enter any URL and get instant risk score
- **🔐 Password Analyzer** — Check password strength with detailed feedback
- **🤖 Security Chatbot** — Ask cybersecurity questions in real-time
- **📊 Project Summary** — Model performance and architecture overview

---

## 🔑 API Key Setup

This project uses the **Groq API** for the AI chatbot (free tier available).

1. Sign up at [console.groq.com](https://console.groq.com)
2. Generate your API key
3. Add it in the Streamlit sidebar or in the notebook cell

---

## 📁 Key Results

```
Module 1 — Phishing URL Detection
  ✅ Test Accuracy  : 96.72%
  ✅ CV Accuracy    : 96.85%
  ✅ ROC-AUC Score  : > 0.98

Module 2 — Password Strength Analyzer
  ✅ Test Accuracy  : 99.87%
  ✅ CV Accuracy    : 99.87% ± 0.04%
  ✅ All 3 classes  : Precision & Recall = 1.00

Module 3 — AI Security Chatbot
  ✅ Model          : LLaMA 3.3 70B
  ✅ Multi-turn     : Yes
  ✅ Response Time  : < 2 seconds
```

---

## 📚 References

- Hannousse, A., & Yahiouche, S. (2021). *Web page phishing detection*. Kaggle.
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Meta AI. (2024). *LLaMA 3.3 70B*. https://ai.meta.com
- Groq Inc. (2024). *Groq API Documentation*. https://console.groq.com

---

## ⚠️ Academic Integrity

This project was developed as part of an Information Security course. All code, models, and analysis are original work. Datasets are publicly available on Kaggle with proper citations.

---

*Information Security Course Project — 2025*
