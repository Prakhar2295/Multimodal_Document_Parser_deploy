import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import requests
import os
import base64


st.title("MULTIMODAL DOCUMENT PARSER")

st.markdown("""
## **Dataset Information : **            
**This dataset is collected from JSL QC Laboratory.These Lab reports are based on the 
material testing reports conducted on various samples stored in the PDF format.
This is a simple Frontend UI for showing prediction results.**""", True)
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">MULTIMODAL DOCUMENT PARSER APPLICATION</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)



file1 = st.file_uploader("Upload a PDF file", type="pdf",key = "NER_EXTRACTION")

if file1 is not None:
    # Read the PDF file

    text = ""
    with pdfplumber.open(file1) as pdf_file:
        for page in pdf_file.pages:
            # Extract text from each page
            text += page.extract_text()
    # Display the content
    #files = {"text": file.getvalue()}
    
    data = {"text": text}
    
    backend_servicename = os.environ.get('BACKEND_SERVICE_NAME')
    #backend_servicename = "http://127.0.0.1:5000"
    response = requests.post(f"{backend_servicename}/predict",json=data)
    
    st.subheader('Please Find the prediction results below')
    
    st.write(f"Prediction: {response.content}")
    

def encode_pdf_to_base64(file_uploader):
    with open(file_uploader.name, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    return encoded_string

    
file2 = st.file_uploader("Upload a PDF file for Table extraction", type="pdf",key="TABLE_EXTRACTION")

if file2 is not None:
    # Read the PDF file
    
    encoded_pdf = encode_pdf_to_base64(file2)
    
    data1 = {"pdf": encoded_pdf}
    
    backend_servicename_layout = os.environ.get('BACKEND_SERVICE_NAME_LAYOUT')
    #backend_servicename = "http://127.0.0.1:5000"
    response_layout = requests.post(f"{backend_servicename_layout}/predict",json=data1)
    
    st.subheader('Please Find the prediction results below')
    
    st.write(f"Layout_Prediction: {response_layout.content}")
    
    