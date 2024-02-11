import streamlit as st
import requests
import os
import base64

st.title("MULTIMODAL DOCUMENT PARSER")

st.markdown("""
## **Dataset Information : **            
**This dataset is collected from JSL QC Laboratory. These Lab reports are based on the 
material testing reports conducted on various samples stored in the PDF format.
This is a simple Frontend UI for showing prediction results.**""", True)
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">MULTIMODAL DOCUMENT PARSER APPLICATION</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Function to encode PDF file as base64 string


file1 = st.file_uploader("Upload a PDF file", type="pdf", key="PDF_UPLOAD")
if file1 is not None:
    # Encode PDF file to base64
    encoded_pdf = base64.b64encode(file1.read()).decode('utf-8')
    print(encoded_pdf)
    
    # Send serialized PDF data to the API
    #backend_servicename = os.environ.get('BACKEND_SERVICE_NAME_LAYOUT')
    backend_servicename = "http://34.222.149.251:5000"
    response = requests.post(f"{backend_servicename}/predict_table", json={"pdf": encoded_pdf})
    
    # Display prediction results
    st.subheader('Please Find the prediction results below')
    st.write(f"Prediction: {response.content}")
