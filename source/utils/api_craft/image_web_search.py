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

# Đường dẫn lưu file feature của clip model
CLIP_FEATURES_PATH=["/dataset/AIC2022/0/CLIPFeatures_C00_V00","/dataset/AIC2022/0/CLIPFeatures_C01_V00","/dataset/AIC2022/0/CLIPFeatures_C02_V00",
                 "/dataset/AIC2022/1/CLIPFeatures_C00_V01","/dataset/AIC2022/1/CLIPFeatures_C01_V01","/dataset/AIC2022/1/CLIPFeatures_C02_V01"] 

KEYFRAME_FOLDER_PATH = "/dataset/AIC2022/" 

# Class chuyển đổi Text -> Vector => code
class TextEmbedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/16", device=self.device)

    def __call__(self, text: str) -> np.ndarray:
        text_inputs = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(text_inputs)[0]
        print(text_inputs)
        return text_feature.detach().cpu().numpy()
   
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
app = flask.Flask("API Text Search")
app.config["DEBUG"] = True
# Load faisss 
db, faiss_db = indexing_methods_faiss(CLIP_FEATURES_PATH)
text_embedd = TextEmbedding()

@app.route('/predict', methods=['POST', 'GET'])
def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    text ="" #câu truy vấn => duong dan anh
    query = "" #thứ tự query
    
    if request.method == "POST":
        query = request.json['query']
        text = request.json['text']

    else:
        query = request.args.get('query')
        text = request.args.get('text')

    # init database 
    global db, faiss_db
    # preprocessing text 
    text_feat_arr = preprocessing_text(text)
    D, I = faiss_db.search(text_feat_arr, k=200)
    search_results = []
    
    for instance in zip(I[0],D[0]):
        ins_id, distance = instance
        video_name, idx= db[ins_id]
        frames_folder = KEYFRAME_FOLDER_PATH + video_name[6] + "/KeyFrames"+ video_name[0:7] +'/'+ video_name
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
def preprocessing_text(text):
    global text_embedd
    text_feat_arr = text_embedd(text) 
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32') #=> float32
    # img_path 
    # load img => (use openCV2)
    # img_vector = img_embedd(img)
    # .... 
    # return img_vector
    return text_feat_arr

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8015, debug=False)