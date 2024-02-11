import streamlit as st
import pdfplumber
import requests
import os
from data_preparation import data_preparation


def main():
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

	if st.button("Please Upload the file for prediction"):
		uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
		st.write("work under progress")

		if uploaded_file is not None:
			original_file_name = uploaded_file.name
			st.write(f"Original File Name: {original_file_name}")
			print(uploaded_file.name)

		if uploaded_file is not None:
			files = uploaded_file.getvalue()
			text = ""
			with pdfplumber.open(files) as pdf_file:
				for page in pdf_file.pages:
					# Extract text from each page
					text += page.extract_text()

			print(text)
			# df =pd.read_csv(csv_file)
			#data = data_preparation(uploaded_file)
			#file = data.pdf_searchable()
			#text_file_dir = "text_file_dir"



			#if file:
				#file_name = data.pdf_to_text()

				#text_file = f"{text_file_dir}/{file_name}"
				#with open(text_file,"rb") as f:
					#text = f.read()
					#text1 = {"text":text}

			#else:
				#data.pdf_to_image()

			#print(text1)
			#files = {uploaded_file.name: uploaded_file.getvalue()}
			# st.dataframe(df)

			# data = df.to_json(orient="records")

			#backend_servicename = os.environ.get('BACKEND_SERVICE_NAME')
			#backend_servicename = "http://127.0.0.1:5000"
			#response = requests.post(f"{backend_servicename}/predict", files=text1)

			#prediction = response.json()["prediction"]
			#st.write(f"prediction: {prediction}")
			#st.write(f"Prediction: {response.content}")


	if st.button("About"):
		st.markdown("""**Built with ❤ by Prakhar**""")


if __name__ == "__main__":
	main()