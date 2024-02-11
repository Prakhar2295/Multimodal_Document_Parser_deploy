import pdfplumber
import pypdfium2 as pdfium
from datetime import datetime
import os

class data_preparation:
	def __init__(self,filename:str):
		self.filename = filename
		self.text_file_path = "text_file_dir"

	def pdf_to_image(self):
		if self.filename is not None:
			pdf = pdfium.PdfDocument(self.filename)
			page = pdf.get_page(0)
			pil_image = page.render(scale=300 / 25).to_pil()
			timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

			#image_name = f"/pdf_to_img/{self.filename[:-4]}.jpg"
			self.image_name = f"/pdf_to_img/{self.filename[:-4]}_{timestamp}.jpg"
			pil_image.save(self.image_name)
			return self.image_name

	def pdf_searchable(self):
		if self.filename is not None:
			text_list = list()
			with pdfplumber.open(self.filename) as pdf:
				page = pdf.pages[0]
				text = page.extract_text()
				text_list.append(text)
			if text_list >= 1:
				return True
			else:
				return False


	def pdf_to_text(self):
		if self.text_file_path is not None:
			pdf = self.pdf_searchable()
			if pdf:
				text = ""
				with pdfplumber.open(self.filename) as pdf_file:
					for page in pdf_file.pages:
						# Extract text from each page
						text += page.extract_text()

				timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

				self.text_file_name = f"{self.filename[:-4]}_{timestamp}.txt"

				self.text_file_full_path = os.path.join(self.text_file_path,self.text_file_name)

				with open(self.text_file_full_path, 'w', encoding='utf-8') as text_file:
					text_file.write(text)

				return self.text_file_name










