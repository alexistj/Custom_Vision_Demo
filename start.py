from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import json
import csv

with open('config.json') as config_file:
    data = json.load(config_file)






ENDPOINT = data['endpoint']

# Replace with a valid key
training_key = data['training_key']
prediction_key = data['prediction_key']
prediction_resource_id = data['prediction_resource_id']

publish_iteration_name = "detectModel"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)


# Find the object detection domain
obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection" and domain.name == "General")

# Create a new project
print ("Creating project...")
project = trainer.create_project("ManTech Vehicle Demo", domain_id=obj_detection_domain.id)



vehicle_tag = trainer.create_tag(project.id, "vehicle")
csv_file = data['csv']

with open(csv_file, mode='r') as values:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
         print(f'Column names are {", ".join(row)}')













