# 🔗 Malicious URL Detector

> AI-powered web application to detect malicious URLs in real time using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Overview

This project is a machine learning web application that classifies URLs as **safe** or **malicious** in real time. It was built to address the growing threat of phishing, malware distribution, and social engineering attacks carried out via malicious links.

The app extracts lexical and structural features from a given URL and passes them through a trained ML model to predict whether the URL poses a threat.

---

## ✨ Features

- 🔍 Real-time URL classification (Safe / Malicious)
- ⚡ Fast predictions powered by a pre-trained XGBoost model
- 🌐 Simple and clean web interface built with Flask
- 📊 96% accuracy on test data
- 🧠 Trained on a large labeled dataset of real-world URLs

---

## 🧠 Models & Approach

| Model | Accuracy |
|-------|----------|
| XGBoost ✅ *(final model)* | **96%** |
| Random Forest | ~94% |

### Feature Engineering

Features extracted from each URL include:

- URL length and structure
- Presence of special characters (`@`, `//`, `-`, etc.)
- Use of IP address instead of domain name
- Number of subdomains
- URL entropy (randomness score)
- HTTPS vs HTTP
- Domain token patterns

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| ML Models | XGBoost, Random Forest (Scikit-learn) |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Model Serialization | Pickle (`.pkl`) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hamdyfahem125/malicious-url-detector.git
cd malicious-url-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

### 4. Open in your browser

```
http://localhost:5000
```

---

## 💡 How It Works

```
User enters URL
      ↓
Feature extraction (length, tokens, special chars, entropy...)
      ↓
Pre-trained XGBoost model
      ↓
Prediction: ✅ Safe  or  ⚠️ Malicious
```

---

## 📁 Project Structure

```
malicious-url-detector/
│
├── app.py              # Flask web application
├── model.pkl           # Pre-trained XGBoost model
├── requirements.txt    # Project dependencies
├── .gitignore
└── README.md
```

---

## 👨‍💻 Author

**Hamdy Ahmed Fahim**
Computer Science Student — AI Track
Higher Technological Institute, Cairo

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://eg.linkedin.com/in/hamdy-fahem-52b74b348)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/hamdyfahem125)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
