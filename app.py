# app.py
from flask import Flask, render_template, request, jsonify
from prediction import *
import json
from PIL import Image
import requests



app = Flask(__name__)
# home route
@app.route("/")
def hello():
    return render_template('index.html', name = 'Jane', gender = 'Female')

# serving form web page
@app.route("/my-form")
def form():
    return render_template('form.html')

app.run(debug = True) 


# handling form data
@app.route('/form-handler', methods=['POST'])
def handle_data():
    
    with open('config.json') as config_file:
        data = json.load(config_file)



    url= str(request.form['URL'])
    ENDPOINT = data['endpoint']
    publish_iteration_name = "detectModel"
    prediction_key = data['prediction_key']
    prediction_resource_id = data['prediction_resource_id']
    training_key = data["training_key"]


 
    try:
        response = requests.get(url)
    except:
        print("error retrieving image: "+url)
        exit(-1)
    
    getPrediction(ENDPOINT, publish_iteration_name, prediction_key, prediction_resource_id, response.content,training_key) 
    