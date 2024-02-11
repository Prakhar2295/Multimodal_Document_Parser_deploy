from flask import Flask,request,Response,jsonify
from model_inference import prediction
from data_preparation import data_preparation
from data_preprocessing import image_preprocessing

app = Flask(__name__)


model_path = "layout_parser_weights/model_final.pth"


@app.route("/",methods = ["POST","GET"])
def raghav():
    return jsonify(message = "Hello world")

@app.route("/predict_table",methods =["GET","POST"])
def predict():
    if request.method == "POST":
        if "pdf" in request.form:
            pdf = request.form["pdf"]
        elif "pdf" in request.json:
            pdf = request.json["pdf"]
        
        else:
            return jsonify(error="No 'text' field provided"), 400
        
        data_prep = data_preparation(pdf)
        image = data_prep.pdf_to_image()
        
        image_prep = image_preprocessing(image)
        _,image = image_prep.check_orientation()
        
        _,image = image_prep.check_skewness()
        
        image = image_prep.remove_watermarks()
        
        df = prediction(image,model_path)
        return Response(f"dataframe: {df}")
    
    


if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5050)
        
        
        
        
        
        
        
        
        
        
        
        




