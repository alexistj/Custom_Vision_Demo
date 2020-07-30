from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import json
from PIL import Image, ImageDraw
import requests
from collections import defaultdict

from datetime import datetime, timedelta
import math
from io import BytesIO
import webbrowser



def drawBounds(img, predictions,):

    # Display the results.
    img = Image.open(BytesIO(img))
    draw = ImageDraw.Draw(img)
    img.save("static/image_results/original.jpg")

   
    img_width = img.size[0]
    img_height = img.size[1]

    crops = []
    counter = 0


    for i, prediction in enumerate(predictions):

        if (prediction.probability*100) > 40:
            left = math.floor(prediction.bounding_box.left * img_width)
            top = math.floor(prediction.bounding_box.top * img_height) 
            height = math.ceil(prediction.bounding_box.height * img_height)
            width =  math.ceil(prediction.bounding_box.width * img_width)

            top_x = left
            top_y = top
            bottom_x = top_x + width
            bottom_y = top_y + height

   

            points = ((left,top), (left+width,top), (left+width,top+height), (left,top+height),(left,top))
            draw.line(points, fill='cyan', width=2)
            counter += 1
            draw.text((left-10, top-10), str(round(prediction.probability*100))+'%', fill='white')
       
    img.save("static/image_results/marked.jpg")
    #webbrowser.open("sample.jpg")

def getPrediction(ENDPOINT, publish_iteration_name, prediction_key, prediction_resource_id, img,training_key,project_name):

    
    # Now there is a trained endpoint that can be used to make a prediction
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    training_credentials   = ApiKeyCredentials(in_headers={"Training-key":training_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
    trainer = CustomVisionTrainingClient(ENDPOINT, training_credentials)
    projects = trainer.get_projects()


    #Retrieve the object dection project and its tags
    #Current assumes one tag
    for p in projects:
        if p.name == project_name:
            project = trainer.get_project(p.id)
            tags = trainer.get_tags(project.id)
            print('Project Found')
   

    # Open the sample image and get back the prediction results.
    results = predictor.detect_image(project.id, publish_iteration_name, img)
    drawBounds(img, results.predictions)


    # Display the results.  
    js_res = defaultdict(list)
    for prediction in results.predictions:
        print(prediction)
        
        x = {
            "confidence":  "{0:.2f}%".format(prediction.probability * 100),
            "bbox_left":   "{0:.2f}".format(prediction.bounding_box.left) ,
            "bbox_right":  "{0:.2f}".format(prediction.bounding_box.top) ,
            "bbox_width":  "{0:.2f}".format( prediction.bounding_box.width),
            "bbox_height": "{0:.2f}".format( prediction.bounding_box.height)
        }

        
        js_res[prediction.tag_name].append(x)
        
   
    return js_res
   

def getPredictionBatch(ENDPOINT, publish_iteration_name, prediction_key, prediction_resource_id, file,training_key,project_name):
    # Now there is a trained endpoint that can be used to make a prediction
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    training_credentials   = ApiKeyCredentials(in_headers={"Training-key":training_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
    trainer = CustomVisionTrainingClient(ENDPOINT, training_credentials)
    projects = trainer.get_projects()



    res_batch ={}
    js_res = {}


    #Retrieve the object dection project and its tags
    #Current assumes one tag
    for p in projects:
        if p.name == project_name:
            project = trainer.get_project(p.id)
            tags = trainer.get_tags(project.id)
            print('Project Found')
   
    for url in file:
        info = url.split()
      
        name = info[0]
        url = info[1]
        try:
            response = requests.get(url)
        except:
            print("error retrieving image: "+url)
            exit(-1)
        # Open the sample image and get back the prediction results.
        results = predictor.detect_image(project.id, publish_iteration_name, response.content)

        # Display the results.  
        js_res["vehicle"] = []
        for prediction in results.predictions:
        
            x = {
                "confidence":  "{0:.2f}%".format(prediction.probability * 100),
                "bbox_left":   "{0:.2f}".format(prediction.bounding_box.left) ,
                "bbox_right":  "{0:.2f}".format(prediction.bounding_box.top) ,
                "bbox_width":  "{0:.2f}".format( prediction.bounding_box.width),
                "bbox_height": "{0:.2f}".format( prediction.bounding_box.height)
            }

            x = json.dumps(x)
            js_res[prediction.tag_name].append(x)
        res_batch[name] = js_res
    
  

    return res_batch




if __name__ == "__main__":
    with open('config.json') as config_file:
        data = json.load(config_file)




    ENDPOINT = data['endpoint']
    publish_iteration_name = "detectModel"
    prediction_key = data['prediction_key']
    prediction_resource_id = data['prediction_resource_id']
    training_key = data["training_key"]
    project_name = data["project_name"]


    url = input('Please insert image url to be tested:')
    try:  
        response = requests.get(url)
    except:
        print("error retrieving image: "+url)
        exit(-1)

    getPrediction(ENDPOINT, publish_iteration_name, prediction_key, prediction_resource_id, response.content,training_key,project_name) 
