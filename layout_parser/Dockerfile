FROM python:3.7.17-slim-bookworm

RUN apt-get update && apt-get -f install && apt-get install -y python3-pip && apt-get install -y git && apt-get install ffmpeg libsm6 libxext6  -y && apt install tesseract-ocr -y && apt install libtesseract-dev -y

RUN mkdir /opt/app

COPY . /opt/app

WORKDIR /opt/app

RUN pip install torch==1.13 torchvision

RUN pip3 install -r requirements.txt

CMD ["python","app.py"]