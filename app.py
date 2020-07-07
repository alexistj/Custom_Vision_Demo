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
    file_name = str(request.form['file_name'])
    ENDPOINT = data['endpoint']
    publish_iteration_name = "detectModel"
    prediction_key = data['prediction_key']
    prediction_resource_id = data['prediction_resource_id']
    training_key = data["training_key"]
    project_name = data["project_name"]


 
    try:
        response = requests.get(url)
    except:
        print("error retrieving image: "+url)
        exit(-1)
    
    js_res  = getPrediction(ENDPOINT, publish_iteration_name, prediction_key, prediction_resource_id, response.content,training_key,project_name) 

    print(js_res)

    
    
    response = app.response_class(
        response=json.dumps(js_res),
        status=200,
        mimetype='application/json'
    )
    return response
    


@app.route('/form-handler-batch', methods=['POST'])
def handle_data_batch():
    
    with open('config.json') as config_file:
        data = json.load(config_file)



    file= request.files['file']
  
    file_name = str(request.form['file_name'])
    ENDPOINT = data['endpoint']
    publish_iteration_name = "detectModel"
    prediction_key = data['prediction_key']
    prediction_resource_id = data['prediction_resource_id']
    training_key = data["training_key"]
    project_name = data["project_name"]





    
    js_res  = getPredictionBatch(ENDPOINT, publish_iteration_name, prediction_key, prediction_resource_id, file,training_key,project_name) 

    print(js_res)

    
    
    response = app.response_class(
        response=js_res,
        status=200,
        mimetype='application/json'
    )
    return response
    

