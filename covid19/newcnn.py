import streamlit as st
from keras_preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import numpy as np
import sqlite3
from io import BytesIO

# Function to initialize SQLite database and create table
def initialize_database():
    conn = sqlite3.connect('patient.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    age INTEGER,
                    phone TEXT,
                    address TEXT,
                    city TEXT,
                    state TEXT,
                    country TEXT,
                    diagnosis TEXT
                )''')
    conn.commit()
    conn.close()

# Function to insert patient data into the database
def insert_patient_data(name, age, phone, address, city, state, country, diagnosis):
    conn = sqlite3.connect('patientdb.db')
    c = conn.cursor()
    c.execute('''INSERT INTO patients (name, age, phone, address, city, state, country, diagnosis) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (name, age, phone, address, city, state, country, diagnosis))
    conn.commit()
    conn.close()

# Load VGG model
vgg16_model = load_model(r"C:\Users\SHIVA\Downloads\vgg16_model.h5")

st.title("COVID-19 DIAGNOSIS AND SEVERITY DETECTION USING CNN and VGG")

# Initialize the SQLite database and create table
initialize_database()

# Patient Information Input Fields
name = st.text_input("Patient Name")
age = st.number_input("Patient Age", min_value=0, max_value=150)
phone = st.text_input("Phone Number")
address = st.text_input("Address")
city = st.text_input("City")
state = st.text_input("State")
country = st.text_input("Country")

# File Uploader for Image
uploaded_file = st.file_uploader("Choose image for diagnosis", type=["jpg", "png", "jpeg"])

def pre():
    if uploaded_file:
        img_path = uploaded_file.read()
        img_path = Image.open(BytesIO(img_path))
        img_path = img_path.convert("RGB")
        img_path = img_path.resize((150, 150))
        image = img_to_array(img_path)
        image = np.expand_dims(image, axis=0)

        # Prediction using VGG model
        vgg_prediction = vgg16_model.predict(image)
        vgg_output = np.argmax(vgg_prediction, axis=1)

        # Mapping diagnosis
        if vgg_output == 0:
            diagnosis = "Mild"
        elif vgg_output == 1:
            diagnosis = "Normal"
        else:
            diagnosis = "Severe"

        # Display Diagnosis
        st.write("Diagnosis: ", diagnosis)

        # Insert patient data into the database
        insert_patient_data(name, age, phone, address, city, state, country, diagnosis)

if st.button("Predict", on_click=pre):
    pre()
