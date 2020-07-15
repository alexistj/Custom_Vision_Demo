# app.py
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, safe_join, abort
from prediction import *
import json
from PIL import Image
import requests
import os


IMAGE_FOLDER = os.path.join('static', 'image_results')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config["DEBUG"] = True




# serving form web page
@app.route("/my-form")
def form():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'placeholder.jpg')
    return render_template('form.html',quicktest_image = full_filename )

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

  
    
    f = open("request/"+file_name+".json", "w")
    f.write(json.dumps(js_res))


    try:
        send_from_directory("request/",filename=file_name+".json", as_attachment=True)
    except FileNotFoundError:
        abort(404)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sample.jpg')
    return render_template("form.html", quicktest_image = full_filename)
    


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

    
    f = open("request/"+file_name+".json", "w")
    f.write(str(js_res))


    try:
        return send_from_directory("request/",filename=file_name+".json", as_attachment=True)
    except FileNotFoundError:
        abort(404)
    



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)