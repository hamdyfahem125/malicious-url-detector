import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

# تحميل النماذج
xgb_model = joblib.load(r"C:\Users\elsha\OneDrive\Documents\AI-URL-Detector\model.pkl")
cnn_model = load_model(r"C:\Users\elsha\OneDrive\Documents\AI-URL-Detector\cnn_url_detector.h5")

# إعداد Character Mapping
chars = 'abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&\'()*+,;=%'
char_to_int = {c: i + 1 for i, c in enumerate(chars)}
max_len = 100

# دالة استخراج الخصائص (XGBoost)
def extract_features(url):
    return {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'has_https': int('https' in url),
        'has_login': int('login' in url),
        'has_secure': int('secure' in url),
        'has_account': int('account' in url),
        'has_update': int('update' in url),
        'has_verify': int('verify' in url)
    }

# دالة ترميز الرابط (CNN)
def encode_url(url):
    return [char_to_int.get(c, 0) for c in url.lower()]

# إعداد الواجهة
st.set_page_config(page_title="Malicious URL Detector", page_icon="🛡️")
st.title("🛡️ Malicious URL Detector")
st.subheader("🔗 Check if a URL is safe or malicious")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=100)
st.sidebar.title("🚨 Project: Malicious URL Detection")
st.sidebar.info("👨‍💻 Developed by: Hamdi Ahmed")

st.markdown("""
> This application detects **malicious URLs** using Artificial Intelligence.
>
> ✅ Choose between **XGBoost Model** (Feature-based) and **CNN Model** (Deep Learning).
""")

# اختيار النموذج
model_choice = st.selectbox("⚙️ Select Detection Model:", ["XGBoost Model", "CNN Model"])

# روابط تجريبية
sample_urls = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "http://secure-login-update.com/account",
    "http://malware-site.com/infected.exe",
    "http://random-unknown-site.xyz",
    "http://cheap-pills-now.com/sale",
    "https://www.youtube.com",
    "http://update-paypal-security.com",
    "http://verify-your-account.com",
]

# زر توليد رابط عشوائي
if st.button("🔄 Try Sample URL"):
    random_url = random.choice(sample_urls)
    st.session_state.sample_url = random_url

# خانة الإدخال
default_url = st.session_state.get("sample_url", "")
user_url = st.text_input("🔗 Enter a URL to check:", value=default_url)

# زر التحقق
if st.button("Check URL") and user_url:
    if model_choice == "XGBoost Model":
        test_features = pd.DataFrame([extract_features(user_url)])
        prediction = xgb_model.predict(test_features)
    else:
        encoded_url = pad_sequences([encode_url(user_url)], maxlen=max_len, padding='post')
        prediction = cnn_model.predict(encoded_url)
        prediction = (prediction > 0.5).astype(int)

    # عرض النتيجة
    if prediction[0] == 1:
        st.error("⚠️ Warning: The URL is Malicious!")
    else:
        st.success("✅ The URL is Safe.")

st.markdown("---")
st.info("📌 Note: This is a demo app for educational purposes.")
