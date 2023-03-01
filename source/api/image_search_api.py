import flask
from flask import jsonify, request
from datetime import datetime
import os 
import numpy as np
from tqdm import tqdm
from PIL import Image
from time import  time
import torch
import sys
sys.path.append("/workspace/sontda/prj/CLIP")
import clip
import faiss
from typing import List, Tuple

#cv2
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Đường dẫn lưu file feature của clip model
CLIP_FEATURES_PATH=["/dataset/AIC2022/0/CLIPFeatures_C00_V00","/dataset/AIC2022/0/CLIPFeatures_C01_V00","/dataset/AIC2022/0/CLIPFeatures_C02_V00",
                 "/dataset/AIC2022/1/CLIPFeatures_C00_V01","/dataset/AIC2022/1/CLIPFeatures_C01_V01","/dataset/AIC2022/1/CLIPFeatures_C02_V01",
                 "/dataset/AIC2022/2/CLIPFeatures_C00_V02","/dataset/AIC2022/2/CLIPFeatures_C01_V02","/dataset/AIC2022/2/CLIPFeatures_C02_V02",
                 "/dataset/AIC2022/2/CLIPFeatures_C00_V03","/dataset/AIC2022/2/CLIPFeatures_C01_V03","/dataset/AIC2022/2/CLIPFeatures_C02_V03",
                 "/dataset/AIC2022/2/CLIPFeatures_C00_V04","/dataset/AIC2022/2/CLIPFeatures_C02_V04"
                 ]

KEYFRAME_FOLDER_PATH = "/dataset/AIC2022/" 

# func chuyển đổi Image -> Vector => code
def ImageEmbedding(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    image_inputs = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_inputs)
    return image_features.detach().cpu().numpy()


# Function load extracted feature into faiss    
def indexing_methods_faiss(clip_features_path):
    faiss_db = faiss.IndexFlatL2(512)
    db = []
    for folder_path in tqdm(clip_features_path):
        # print(folder_path)
        for feat_npy in tqdm(os.listdir(folder_path)):
            # print(feat_npy)
            video_name = feat_npy.split('.')[0]
            feats_arr = np.load(os.path.join(folder_path, feat_npy))
            for idx, feat in enumerate(feats_arr):
            #Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes
                instance = (video_name, idx)
                db.append(instance)
                faiss_db.add(feat.reshape(1,-1).astype('float32'))
    db = dict(enumerate(db))
    return db, faiss_db

# Gọi lên chạy 
app = flask.Flask("Image Search")
app.config["DEBUG"] = True

# Load faisss 
db, faiss_db = indexing_methods_faiss(CLIP_FEATURES_PATH)

@app.route('/predict', methods=['POST', 'GET'])
def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    
    if request.method == "POST":
        query = request.json['query']
        image_url = request.json['image_url']

    else:
        query = request.args.get('query')
        image_url = request.args.get('image_url')


    # init database 
    global db, faiss_db

    # preprocessing image
    img_vector = preprocess_image(image_url)

    D, I = faiss_db.search(img_vector, k=200)

    search_results = []
    
    for instance in zip(I[0],D[0]):
        ins_id, distance = instance
        video_name, idx= db[ins_id]
        in_folder = video_name[6]
        if (in_folder>'2'):
            in_folder='2'
        frames_folder = KEYFRAME_FOLDER_PATH + in_folder + "/KeyFrames"+ video_name[0:7] +'/'+ video_name
        keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
        video_name = video_name + '.mp4'
        result = {"video_name":str(video_name),
                                "keyframe_id": str(keyframe_id),
                                "score": str(distance)}
        print("result: ", result)
        search_results.append(result)
    
    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  


#preprocess vector đặc trưng của image (truyền vào đường dẫn của image)
def preprocess_image(image_url):
    image_url = image_url.lstrip()
    if (image_url[0:4]=="http"):
        response = requests.get(image_url)
        image_path = BytesIO(response.content)
    else:
        image_path=image_url
    img_vector = ImageEmbedding(image_path)
    img_vector = img_vector.reshape(1,-1).astype('float32')
    return img_vector

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8019, debug=False)