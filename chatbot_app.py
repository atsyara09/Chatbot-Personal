import streamlit as st
import re
import os
import base64
import random
import pickle
import json
import time
import string
import numpy as np
from fpdf import FPDF 
from nltk.corpus import stopwords
from datetime import datetime
from tensorflow.keras.models import load_model
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from db_connection import get_connection

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

@st.cache(allow_output_mutation=True)
def load_intents():
    with open("data/intents.json") as f:
        return json.load(f)

data = load_intents()

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as f:
        return pickle.load(f)
    
tokenizer = load_tokenizer()

@st.cache(allow_output_mutation=True)
def load_label_encoder():
    with open("label_encoder.pickle", "rb") as f:
        return pickle.load(f)
    
le = load_label_encoder()

@st.cache(allow_output_mutation=True)
def load_chatbot_model():
    return load_model("chatbot_model.h5")
    
model = load_chatbot_model()
input_length = model.input_shape[1]

def preprocess(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)


def get_base64_img(path):
    with open(path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
        return f"data:image/png;base64,{b64}"

bot_avatar = get_base64_img("images/bot_avatar.jpeg")
user_avatar = get_base64_img("images/user_avatar.jpeg")
logo_kampus = get_base64_img("images/logo_kampus.jpg")

st.markdown(f"""
    <div style='display: flex; align-items: center; margin-bottom: 15px;'>
        <img src="{get_base64_img("images/logo_kampus.jpg")}" width="80" style="margin-right: 20px; border-radius: 8px;" />
        <div>
            <h2 style='margin-bottom: 5px;'>Universitas Ichsan Gorontalo</h2>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"""
    <style>        
    .chat-container {{
        display: flex;
        flex-direction: column;
        margin-top: 20px;
    }}
    .chat-row {{
        display: flex;
        align-items: flex-end;
        margin-bottom: 12px;
    }}
    .chat-bubble {{
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 12px;
        font-size: 16px;
        line-height: 1.4;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }}
    .bot-message {{
        background-color: #e1f7d5;
        border-top-left-radius: 0;
        margin-left: 10px;       
    }}
    .user-message {{
        background-color: #ffffff;
        border-top-right-radius: 0;
        border: 1px solid #ddd;
        margin-right: 10px;
    }}
    .avatar {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
    }}
    (st.form_submit_button)
    div.stButton > button {{
        background-color: #e1f7d5;
        color:white;
        border-radius:25px;
        padding:12px 25px;
        border:none;
        box-shadow:0px 4px 10px rgba(0,0,0,0.15);
        font-weight:bold;
        transition:all 0.3s ease;
        width:100%;
    }}
    div.stButton > button:hover{{
        background-color: #e1f7d5;
        transform:translateY(-2px);        
    }}
    .stDownloadButton > 
    button,.stDownloadButton > button:focus{{
        background-color: #ffffff;
        color:black;
        border-radius:25px;
        padding: 12px 20px;
        border:none;
        font-weight:normal;
        transition:all 0.3s ease;
        width: 100%;      
    }} 
    .stDownloadButton > button:hover{{
        background-color: #e1f7d5;
        transform:translateY(-2px);
    }}
    .st-emotion-cache-ue6h4q,.st-emotion-cache-12152lb{{
        background-color: #f0f0f0;
        border-radius:10px;
        padding:10px 15px;
        margin-top:20px;
        
    }}
    div.stButton[data-testid="stFormSubmitButton"] 
    + div.stButton > button{{
        background-color: #dc3545;
        color:white;
        border-radius:25px;
        padding:12px 25px;
        border:none;
        box-shadow:0px 4px 10px rgba(0,0,0,0.15);
        font-weight:bold;
        transition:all 0.3s ease;
        margin-top:15px;
        width: 100%; 
    }}
    div.stButton[data-testid="stFormSubmitButton"] 
    + div.stButton > button:hover{{
        background-color: #c82333;
        transform:translateY(-2px);
    }}
    .left {{ justify-content: flex-start; }}
    .right {{ justify-content: flex-end; flex-direction: row-reverse; }}
    @media only screen and (max-width: 600px) {{
        .chat-bubble {{
            max-width: 85%;
            font-size: 14px;
            padding: 8px; 12px;
        }}
        .chat-row {{
            flex-wrap: wrap;
        }}
        .avatar {{
            width: 30px;
            height: 30px;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

st.title("AI Chatbot Layanan Akademik")
st.markdown("---") 

def initialize_chat():
    st.session_state.history = [("bot", "Halo! Saya adalah Chatbot Layanan Akademik. Ada yang bisa saya bantu terkait layanan akademik?")]

if "history" not in st.session_state:
    initialize_chat()
    
def get_response(user_input):
    user_input_clean = preprocess(user_input)
    seq = tokenizer.texts_to_sequences([user_input_clean])
    pad = pad_sequences(seq, maxlen=input_length, padding='post')
    prob = model.predict(pad)[0] 
    confidence = np.max(prob)
    intent = le.classes_[np.argmax(prob)]

    save_message("user", user_input)

    if confidence < 0.85:
        bot_reply = "Maaf, saya tidak paham maksud Anda."
    else:
        for tg in data['intents']:
            if tg['tag'] == intent:
                bot_reply = random.choice(tg['responses'])
                break
        else:
            bot_reply = "Maaf, terjadi kesalahan."

    save_message("bot", bot_reply, intent=intent)
    return bot_reply

def save_message(sender, message, intent="unknown"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history (sender, message, intent, timestamp) VALUES (%s, %s, %s, %s)",
        (sender, message, intent, datetime.now())
    )
    conn.commit()
    cur.close()
    conn.close()
    
def render_whatsapp_chat_with_avatar():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for sender, msg in st.session_state.history:
        if sender == "bot":
            avatar = bot_avatar
            align = "left"
            bubble_class = "bot-message"
        else:
            avatar = user_avatar
            align = "right"
            bubble_class = "user-message"

        st.markdown(f"""
            <div class="chat-row {align}">
                <img src="{avatar}" class="avatar">
                <div class="chat-bubble {bubble_class}">{msg}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def generate_chat_text():
    chat_text = ""
    for sender, msg in st.session_state.history:
        name = "Anda" if sender == "user" else "Bot"
        chat_text += f"{name}: {msg}\n"
    return chat_text

def sanitize_text(text):
    return re.sub(r'[^\x00-\x7F\u00A0-\uFFFF]', '', text)
    
def export_pdf(chat_text):
    pdf = FPDF()
    try:
        pdf.add_font('NotoSans', '', 'font/NotoSans-Regular.ttf', uni=True)
        pdf.set_font('NotoSans', '', 12)
    except Exception as e:
        print(f"[WARNING] Gagal load font NotoSans, fallback ke Arial: {e}")
        pdf.set_font('Arial', size=12)

    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    for i, line in enumerate(chat_text.split("\n"), start=1):
        clean_line = sanitize_text(line.strip())

        if not clean_line or (len(clean_line) > 200 and " " not in clean_line):
            print(f"[SKIP] Baris {i} kosong atau terlalu panjang tanpa spasi: {repr(line)}")
            continue

        try:
            pdf.multi_cell(0, 10, clean_line)
        except Exception as e:
            print(f"[ERROR] Gagal render baris {i}: {repr(clean_line)}")
            pdf.multi_cell(0, 10, "[Baris tidak dapat ditampilkan]")

    pdf_data = pdf.output(dest='S')
    return pdf_data.encode('latin-1') if isinstance(pdf_data, str) else pdf_data

render_whatsapp_chat_with_avatar()
    
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ketik pesan Anda...", "") 
    submit = st.form_submit_button("Kirim")

    if submit and user_input:
        st.session_state.history.append(("user", user_input))

        with st.spinner("Bot sedang berpikir..."):
            bot_reply = get_response(user_input) 
            time.sleep(min(0.5, len(bot_reply) * 0.02)) 
        st.session_state.history.append(("bot", bot_reply))
        st.experimental_rerun()
        

st.markdown("---")
with st.expander("Simpan atau Ekspor Percakapan"):
    chat_text = generate_chat_text()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download TXT",
            data=chat_text,
            file_name="percakapan_chatbot.txt",
            mime="text/plain"
        )
    with col2:
        if chat_text.strip(): 
            pdf_bytes = export_pdf(chat_text)
            st.download_button(
                label="Download PDF",
                data=export_pdf(chat_text),
                file_name="percakapan_chatbot.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Ketik pesan untuk memulai percakapan sebelum mengekspor.")

st.markdown(
    """
    <style>
    div.stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True)

if st.button("Mulai Percakapan Baru"): 
   initialize_chat()
   st.experimental_rerun()

   
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 30px;'>Made with ❤️ by Me Namaste</p>", unsafe_allow_html=True)