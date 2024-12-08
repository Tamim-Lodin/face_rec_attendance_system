import numpy as np
import pandas as pd
import cv2

import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Connect to Redis Client
hostname = ''
portnumber = 19001
password = ''

r = redis. StrictRedis(host=hostname,
port=portnumber,
password=password)
# configure face analysis
from insightface.app import FaceAnalysis
# Configure face analysis from insightface models
faceapp = FaceAnalysis(name='buffalo_sc', 
                       root='insightface_models', 
                       providers=['CPUExecutionProvider'])

# Preparing face analysis with specified context and detection size
faceapp.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)  
# ML search algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector,name_role=['name','role'],thresh=0.5):

#cosine similarity base search algorithm
    # Assuming X_list is a list of embedding vectors
    x = np.vstack(x_list)  # Stack all embeddings into a 2D array
    
    
    # Ensure the test_vector is reshaped properly for cosine similarity
    similar = pairwise.cosine_similarity(X, test_vector.reshape(1, -1))
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # step-2: Index face embeding from the dataframe and convert into array
    x_list = dataframe[feature_column].tolist()
    x = np.asarray(x_list)
    
    # step-3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr
    
    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine' ].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
    return person_name, person_role
def face_prediction(test_image,dataframe, feature_column,name_role=['name','role'],thresh=0.5):
    # step-1: take the test image and apply to insight face
    results = faceapp.get(test_image)
    test_copy = test_image.copy()
    # step-2: use for loop and extract each embedding and pass to ml_search_algorithm
    
    for res in results:
        x1, y1,x2,y2=res['bbox'].astype(int)
        embeddings = res['embedding' ]
        # testing with both the dataset and compressed dataset which obtained in below cell
        #person_name, person_role = ml_search_algorithm(dataframe,'facial_features',test_vector=embeddings,name_role=['name','role'],thresh=0.5)
        person_name, person_role = ml_search_algorithm(dataframe,feature_column,test_vector=embeddings,name_role=name_role,thresh=thresh)
        
        
        
        if person_name == 'Unknown':
            color =(0,0,255) # bgr
        else:
            color = (0,255,0)
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
        text_gen = person_name
        cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
    return test_copy

