#Developed by Daniel Cyrus
import sys
import os

sys.path.append("models/research/")
sys.path.append("models/")

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file("ssd8/pipeline.config")
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("ssd8/", 'ckpt-51')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt", use_display_name=True)


from cv2 import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import matplotlib.pyplot as plt


def checkColor(image):
    image = cv2.resize(image,(50,50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #mask = cv2.inRange(image, (36, 30, 30), (85, 255,255))
    #temp = cv2.bitwise_and(image,image, mask= mask)
    #image = image - temp

    #image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    hsvrange = {"red":range(28),"yellow":range(28,36),"green":range(36,85) ,"blue":range(85,135),"magenda":range(135,160),"red1":range(160,180)}
    
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    # Find and display most dominant colors
    cluster = KMeans(n_clusters=10).fit(reshape)
    
    
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    colours = {"red":0.0,"yellow":0.0,"green":0.0 ,"blue":0.0,"magenda":0.0,"red1":0.0,"black":0.0, "white":0.0}
    for (percent, colour) in zip(hist, cluster.cluster_centers_):
      #print(colour ,"\t {0:.2f}%".format(percent *100))
      for key in hsvrange:
        if((int(colour[0]) in hsvrange[key]) and colour[1]>=90 and colour[2]>=60):
          colours[key] += percent
        elif(colour[1]<90 and colour[2]>=60):
          colours["white"] += percent
        elif(colour[2]<60):
          colours["black"] += percent


    val = 0.0
    _key=""
    if (colours["red"]>0.01 or colours["yellow"]>0.01) and colours["black"]< 0.02:
      colours["red"]+=colours["red1"]+colours["yellow"]+colours["white"]
    elif (colours["red"]>0.01 or colours["yellow"]>0.01) and colours["black"]>= 0.02: 
      colours["yellow"]+=colours["red"]+colours["black"]
    else:
      colours["blue"]+=colours["black"]
    retrunVal = {"red":(0,0,255),"yellow":(0,255,255),"green":(0,255,0) ,"blue":(255,0,0),"magenda":(255,0,0),"red1":(0,0,255),"black":(255,0,0), "white":(0,0,255)}  
    for key in colours:
      if(key!="green" and colours[key]>val):
        val=colours[key]
        _key = key

    if colours["red"]==0 and colours["yellow"]==0 and colours["blue"]==0 and colours["magenda"]==0 and colours["red1"]==0 and colours["white"]==0:return (255,0,0)
    return retrunVal[_key]



from cv2 import cv2
import json
from django.forms.models import model_to_dict

cap  = cv2.VideoCapture("match_calibrated.avi")
#---------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vout = cv2.VideoWriter('18Jan_ssd.mp4',fourcc, 30, (2000,898))
#---------------------------------------
dataPos=[]          
#--------------------------------------
import numpy as np
from pathlib import Path
sec=0
while cap.isOpened():
    ret, image_np = cap.read()
    if image_np is not None:
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      #image_np_expanded = np.expand_dims(image_np, axis=0)

      input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
      detections, predictions_dict, shapes = detect_fn(input_tensor)

      label_id_offset = 1
      image_np_with_detections = image_np.copy()
      '''
      viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=300,
          min_score_thresh=.40,
          agnostic_mode=False)
      

      '''
      positions=np.squeeze(detections['detection_boxes'][0])
      pclass=np.squeeze(detections['detection_classes'][0] + label_id_offset)
      scores=detections['detection_scores'][0].numpy()
           
      ppos=[]
      bpos=[]
      colours=[]
      
      newImg= image_np.copy()
      for _i, pc in enumerate(pclass):
        if pc==1.0 and scores[_i] > .4 :
          
          x = int(positions[_i][1] * image_np.shape[1])
          y = int(positions[_i][0] * image_np.shape[0])

          x2 = int(positions[_i][3] * image_np.shape[1])
          y2 = int(positions[_i][2] * image_np.shape[0])
          bndY = int((y2-y)/5)
          bndX = int((x2-x)/5)
          crp = image_np[y+bndY:y2-bndY, x+bndX:x2-bndX].copy()
          box_color = checkColor(crp)
         
          ppos.append(np.array(positions[_i]).tolist())
          colours.append(np.array(box_color).tolist())
          newImg = cv2.rectangle(newImg, (x,y),(x2,y2),(box_color),2)

          #print(box_color)
          
        if pc==2.0 and scores[_i] > .4 :
          bpos=positions[_i].tolist()
          x = int(positions[_i][1] * image_np.shape[1])
          y = int(positions[_i][0] * image_np.shape[0])

          x2 = int(positions[_i][3] * image_np.shape[1])
          y2 = int(positions[_i][2] * image_np.shape[0])
          colours.append(np.array(box_color).tolist())
          newImg = cv2.circle(newImg,(x,y), 5, (100,255,0), 2)
      #print(pcount)
      dataPos.append({"poses":ppos,"colours":colours ,"ball":bpos})
      
      
      #print((detections['detection_classes'][0].numpy() + label_id_offset).astype(int))
      
      # Display output
      vout.write(newImg)
      sec=(int)(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
      print(sec)
      
      
      #cv2.imshow('player detection',cv2.resize(image_np_with_detections, (980, 250)))
#print(dataPos)    
with open('positions_ssd.json', 'w') as outfile:
          json.dump(dataPos, outfile)
vout.release()
cap.release()
print("Done")
#cv2.destroyAllWindows()
