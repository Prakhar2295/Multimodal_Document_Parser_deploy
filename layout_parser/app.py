from flask import Flask,request,Response,jsonify
from model_inference import prediction
from data_preparation import data_preparation
from data_preprocessing import image_preprocessing
import base64
import cv2

app = Flask(__name__)


model_path = "layout_parser_weights/model_final.pth"



@app.route("/",methods = ["POST","GET"])
def raghav():
    return jsonify(message = "Hello world")

@app.route("/predict_table",methods =["GET","POST"])
def predict():
    if request.method == "POST":
        if "pdf" in request.form:
            pdfData = request.form["pdf"]
        elif request.json and "pdf" in request.json:
            pdfData = request.json["pdf"]
        
        else:
            return jsonify(error="No 'text' field provided"), 400
        
        decodedData = base64.b64decode((pdfData))
        
        filename = 'output.pdf'
        pdfFile = open(filename, 'wb')
        pdfFile.write(decodedData)
        pdfFile.close()
        
        f = open("logging.txt","w")
        
        
        data_prep = data_preparation(filename)
        image = data_prep.pdf_to_image()
        img = cv2.imread(image)
        f.write(f"pdf2img: {img.shape}")
        
        image_prep = image_preprocessing(image)
        #image = image_prep.check_orientation()
        
        #image = image_prep.check_skewness()
        
        image = image_prep.remove_watermarks()
        img = cv2.imread(image)
        f.write(f"remove_watermarks: {img.shape}")
        
        
        df = prediction(image,model_path)
        #f.write(f"final_dataframe: {df.shape}")
        f.close()
        #return Response(f"dataframe: {df}")
        return jsonify(data=df)
    
    


if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5050)
        
        
        
        
        
        
        
        
        
        
        
        




