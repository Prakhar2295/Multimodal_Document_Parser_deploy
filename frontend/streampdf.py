import streamlit as st
import pdfplumber
#import PyPDF2
import requests
import os


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



file = st.file_uploader("Upload a PDF file", type="pdf")

if file is not None:
    # Read the PDF file

    text = ""
    with pdfplumber.open(file) as pdf_file:
        for page in pdf_file.pages:
            # Extract text from each page
            text += page.extract_text()
    # Display the content
    #files = {"text": file.getvalue()}
    
    data = {"text": text}
    
    #backend_servicename = os.environ.get('BACKEND_SERVICE_NAME')
    backend_servicename = "http://127.0.0.1:5000"
    response = requests.post(f"{backend_servicename}/predict",json=data)
    
    st.subheader('Please Find the prediction results below')
    
    st.write(f"Prediction: {response.content}")