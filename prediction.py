from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import json
from PIL import Image
import requests




with open('config.json') as config_file:
    data = json.load(config_file)






ENDPOINT = data['endpoint']
publish_iteration_name = "detectModel"
prediction_key = data['prediction_key']
prediction_resource_id = data['prediction_resource_id']




# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
training_credentials = ApiKeyCredentials(in_headers={"Training-key": data["training_key"]})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
trainer = CustomVisionTrainingClient(data["endpoint"], training_credentials)
projects = trainer.get_projects()


#Retrieve the object dection project and its tags
#Current assumes one tag
for p in projects:
    if p.name == data["project_name"]:
        project = trainer.get_project(p.id)
        tags = trainer.get_tags(project.id)
        print('Project Found')
    


url = input('Please insert image url to be tested:')
try:
    response = requests.get(url)
except:
    print("error retrieving image: "+url)
    exit(-1)


# Open the sample image and get back the prediction results.
results = predictor.detect_image(project.id, publish_iteration_name, response.content)

# Display the results.    
for prediction in results.predictions:
    print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))