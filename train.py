from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import json
import csv
from collections import defaultdict 
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import time
from past.builtins import xrange



with open('config.json') as config_file:
    data = json.load(config_file)


csv_file = data['csv']



credentials = ApiKeyCredentials(in_headers={"Training-key": data["training_key"]})
trainer = CustomVisionTrainingClient(data["endpoint"], credentials)
publish_iteration_name = "detectModel"
projects = trainer.get_projects()
prediction_resource_id = data['prediction_resource_id']
tagged_images_with_regions = []
vehicle_regions = defaultdict(list)

#Retrieve the object dection project and its tags
#Current assumes one tag
for p in projects:
    if p.name == data["project_name"]:
        project = trainer.get_project(p.id)
        tags = trainer.get_tags(project.id)
        print('Project Found')
    




print('Loading locations')
with open(csv_file, 'r',encoding='utf-8' ) as csv_f:
    reader = csv.reader(csv_f)
    line_count = 0
    
  
    d= []

    #Parse the images regions given by csv and store in a dictionary
    
    for row in reader:
        if line_count == 0:
            line_count+=1
            continue
        x = row[11].replace('[','')
        x = x.replace(']','')
        d = json.loads(x)
        d['url'] = row[1]
        vehicle_regions[row[2]].append(d)
    
    

    #Applying labels to images provided by csv 
    print('Generating regions')
    for image in vehicle_regions:
        url = vehicle_regions[image][0]['url']
        try:
            response = requests.get(url)
            print('recieved')
        except:
            print("error retrieving image: "+url)
            continue
        img = Image.open(BytesIO(response.content))
        regions =  []
        region_count =0
        for region in vehicle_regions[image]:
            if region_count >= 200:
                break
            regions.append( Region(tag_id=tags[0].id, left=float(region['coordinates']['x']),top=float(region['coordinates']['y']),  width=float(region['coordinates']['w']),height=float(region['coordinates']['h']))   )
            region_count+=1
        tagged_images_with_regions.append(ImageFileCreateEntry(name=image, contents=response.content, regions=regions))
      
         
    

#uploading labeled images to custom vision portal
print('Uploading images to trainer')
for i in range(0,len(tagged_images_with_regions),63):
    batch =  tagged_images_with_regions[i:i+63]
    upload_result = trainer.create_images_from_files(project.id, images=batch)
    if not upload_result.is_batch_successful:
       print("Image batch upload failed.")
       for image in upload_result.images:
            print("Image status: ", image.status)
       exit(-1)


#Training and creating new itteration of model 
print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")
