# fundbuero_app_complete.py
"""
Digitales Schul-Fundbüro mit Teachable Machine
================================================
Features:
- Jeder Kategorie ein eigener Tab in der Fundübersicht
- Große Vorschaubilder für einfache Übersicht
- Fundsachen klassifizieren über Teachable Machine Keras-Modell
- Speicherung von Bildern, Funddatum und Abholstatus
- Bildaufnahme über Kamera oder Upload von Datei
"""

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import json
from datetime import datetime
from tensorflow.keras.models import load_model

# ---------------------------
# 1. Globale Einstellungen
# ---------------------------

st.set_page_config(
    page_title="Schul-Fundbüro",
    page_icon="🧸",
    layout="wide"
)

# Verzeichnisse
DATA_DIR = "fundsachen_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DB_FILE = os.path.join(DATA_DIR, "fundsachen.json")
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

# Sicherstellen, dass Verzeichnisse existieren
os.makedirs(IMAGE_DIR, exist_ok=True)
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ---------------------------
# 2. Hilfsfunktionen
# ---------------------------

@st.cache_resource
def lade_ki_modell():
    """Lädt das Teachable Machine Keras-Modell und die Labels"""
    model = load_model(MODEL_PATH, compile=False)
    labels = [line.strip() for line in open(LABELS_PATH, "r", encoding="utf-8")]
    return model, labels

def klassifiziere_bild(img: Image.Image, model, labels):
    """Klassifiziert ein Bild mit dem Keras-Modell"""
    img_resized = ImageOps.fit(img.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized, dtype=np.float32)
    normalized_array = (img_array / 127.5) - 1
    data = np.expand_dims(normalized_array, axis=0)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return labels[index], prediction[0][index]

def lade_db():
    """Lädt die JSON-Datenbank"""
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def speichere_db(db):
    """Speichert die JSON-Datenbank"""
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=4)

def speichere_bild(img: Image.Image, filename: str):
    """Speichert das Bild im IMAGE_DIR"""
    path = os.path.join(IMAGE_DIR, filename)
    img.save(path)
    return path

# ---------------------------
# 3. App-Navigation
# ---------------------------

model, labels = lade_ki_modell()

st.sidebar.title("Navigation")
seite = st.sidebar.radio("Bereich wählen:", ["Fundsache melden", "Fundübersicht"])

# ===========================
# Bereich: Fundsache melden
# ===========================
if seite == "Fundsache melden":
    st.title("📸 Fundsache melden")

    st.info("Wähle aus, wie du die Fundsache erfassen möchtest:")

    option = st.radio("Bildquelle auswählen:", ["Kamera", "Datei hochladen"])
    img = None

    if option == "Kamera":
        uploaded_image = st.camera_input("Foto aufnehmen")
        if uploaded_image is not None:
            img = Image.open(uploaded_image)

    elif option == "Datei hochladen":
        uploaded_file = st.file_uploader("Bild der Fundsache hochladen", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)

    if img is not None:
        st.image(img, caption="Erfasstes Bild", use_column_width=True)

        klasse, confidence = klassifiziere_bild(img, model, labels)
        st.success(f"Die KI erkennt diese Fundsache als: **{klasse}** ({confidence*100:.2f}%)")

        if st.button("Fundsache speichern"):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            dateiname = f"{timestamp}.png"
            speichere_bild(img, dateiname)

            db = lade_db()
            eintrag = {
                "dateiname": dateiname,
                "kategorie": klasse,
                "funddatum": datetime.now().strftime("%d.%m.%Y"),
                "status": "nicht abgeholt"
            }
            db.append(eintrag)
            speichere_db(db)
            st.success("Fundsache erfolgreich gespeichert!")

# ===========================
# Bereich: Fundübersicht (Tabs)
# ===========================
elif seite == "Fundübersicht":
    st.title("🗂️ Fundübersicht")
    db = lade_db()
    zeige_abgeholt = st.checkbox("Abgeholte Gegenstände anzeigen", value=True)

    tabs = st.tabs(labels)
    for idx, kategorie in enumerate(labels):
        with tabs[idx]:
            st.subheader(kategorie.capitalize())
            kategorie_eintraege = [e for e in db if e["kategorie"] == kategorie]
            if not zeige_abgeholt:
                kategorie_eintraege = [e for e in kategorie_eintraege if e["status"] != "abgeholt"]

            if kategorie_eintraege:
                for eintrag in kategorie_eintraege:
                    bildpfad = os.path.join(IMAGE_DIR, eintrag["dateiname"])
                    img = Image.open(bildpfad)
                    caption = f"{eintrag['funddatum']} | {'✅ Abgeholt' if eintrag['status']=='abgeholt' else '⬜ Nicht abgeholt'}"
                    st.image(img, caption=caption, use_column_width=True)

                    if eintrag["status"] != "abgeholt":
                        if st.button(f"Als abgeholt markieren ({eintrag['dateiname']})"):
                            eintrag["status"] = "abgeholt"
                            speichere_db(db)
                            st.experimental_rerun()
            else:
                st.info("Keine Fundsachen in dieser Kategorie.")
