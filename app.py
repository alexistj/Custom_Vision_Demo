# app.py
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, safe_join, abort
from prediction import *
import json
from PIL import Image
import requests
import os


IMAGE_FOLDER = os.path.join('static', 'image_results')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config["DEBUG"] = True

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


# serving form web page
@app.route("/my-form")
def form():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'placeholder.jpg')
    return render_template('form.html',quicktest_image = full_filename , display = "display: none;")

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
    project_name = data["project_name"]


 
    try:
        response = requests.get(url)
    except:
        print("error retrieving image: "+url)
        exit(-1)
    
    js_res  = getPrediction(ENDPOINT, publish_iteration_name, prediction_key, prediction_resource_id, response.content,training_key,project_name) 
    js_res =json.dumps(js_res, indent=4, separators=(". ", " = "))
 
    #f = open("request/"+"quicktest_image"+".json", "w")
    #f.write(json.dumps(js_res))


    #try:
     #   send_from_directory("request/",filename=file_name+".json", as_attachment=True)
    #except FileNotFoundError:
    #    abort(404)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg')
    return render_template("form.html", quicktest_image = full_filename, js_res = js_res, display = ";")
    


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