from PIL import Image
import pytesseract
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import math
import time
import sys


class image_preprocessing:
	def __init__(self,filename:str):
		self.filename = filename

	def check_orientation(self):
		if self.filename is not None:
			im = Image.open(self.filename)
			osd = pytesseract.image_to_osd(im, output_type='dict')
			rotate = osd['rotate']

			if not os.path.isdir("rotated_img_dir"):
				os.mkdir("rotated_img_dir")

			if rotate > 0:
				timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
				im_fixed = im.copy().rotate(rotate)
				image_name = f"/rotated_img_dir/self.filename_rotated_{timestamp}.jpg"
				im_fixed.save(image_name)
				return True,image_name
			else:
				return False

	def check_skewness(self):
		if self.filename is not None:
			img = cv2.imread(self.filename)
			img_shape = img.shape
			time1 = time.time()
			# gray
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# blank space to draw houghlines
			blank = np.uint8(np.zeros((img_shape[0], img_shape[1])))
			skernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

			# binarization
			ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			# denoise
			opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, skernel)
			# erosion
			erosion = cv2.erode(opening, skernel)
			# canny edge detection
			edges = cv2.Canny(erosion, 80, 240, apertureSize=3)
			# hough line transform
			lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=250, maxLineGap=70)
			lines1 = lines[:, 0, :]
			Theta = []
			for x1, y1, x2, y2 in lines1[:]:
				cv2.line(blank, (x1, y1), (x2, y2), (255, 255, 255), 5)
				# calculate angle
				theta = math.atan2(y1 - y2, x2 - x1)
				Theta.append(theta * 180 / np.pi)
			# find the most angle
			angle_i = np.histogram(Theta, bins=90)[0].tolist()
			angle_m = angle_i.index(max(angle_i))
			angle = np.histogram(Theta, bins=90)[1].tolist()
			# rotate
			rows, cols = img.shape[:2]
			angle_i = np.histogram(Theta, bins=90)[0].tolist()
			angle_m = angle_i.index(max(angle_i))
			angle = np.histogram(Theta, bins=90)[1].tolist()
			height, width = img.shape[:2]
			degree = angle[angle_m] * -1
			heightNew = int(
				width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
			widthNew = int(
				height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))
			M = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
			M[0, 2] += (widthNew - width) / 2
			M[1, 2] += (heightNew - height) / 2
			res = cv2.warpAffine(img, M, (widthNew, heightNew))

			if not os.path.isdir("skew_img_dir"):
				os.mkdir("skew_img_dir")

			if angle[angle_m] > 0:
				timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
				image_name = f"/skew_img_dir/self.filename_rotated_{timestamp}.jpg"
				cv2.imwrite(image_name,res)
				return True,image_name
			else:
				return False

	def remove_watermarks(self):
		if self.filename is not None:
			img = cv2.imread(self.filename)
			# Convert the image to grayscale
			gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# Make a copy of the grayscale image
			bg = gr.copy()
			# Apply morphological transformations
			for i in range(5):
				kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
				                                    (2 * i + 1, 2 * i + 1))
				bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel2)
				bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel2)
			# Subtract the grayscale image from its processed copy
			dif = cv2.subtract(bg, gr)
			# Apply thresholding
			bw = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			dark = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			# Extract pixels in the dark region
			darkpix = gr[np.where(dark > 0)]
			# Threshold the dark region to get the darker pixels inside it
			darkpix = cv2.threshold(darkpix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			# Paste the extracted darker pixels in the watermark region
			bw[np.where(dark > 0)] = darkpix.T
			timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

			if not os.path.isdir("background_preprocess_dir"):
				os.mkdir("background_preprocess_dir")


			image_name = f"/background_preprocess_dir/self.filename_rotated_{timestamp}.jpg"
			cv2.imwrite(image_name,bw)
			return image_name


	def document_dewarping(self):
		pass






