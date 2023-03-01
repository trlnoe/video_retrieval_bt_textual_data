import flask
from flask import jsonify, request
from datetime import datetime
import os
import numpy as np
import json
from tqdm import tqdm
from time import  time
import torch
import sys
sys.path.append("/workspace/sontda/prj/CLIP")
print("sys path: ", sys.path) 

import clip
import faiss

# sys.path.append("/workspace/sontda/prj/CLIP")
# print("sys path: ", sys.path)

### STATIC VALUE
CLIP_FEATURES_PATH = {
    "B16": ["/dataset/AIC2022/0/CLIPFeatures_C00_V00","/dataset/AIC2022/0/CLIPFeatures_C01_V00","/dataset/AIC2022/0/CLIPFeatures_C02_V00",
                 "/dataset/AIC2022/1/CLIPFeatures_C00_V01","/dataset/AIC2022/1/CLIPFeatures_C01_V01","/dataset/AIC2022/1/CLIPFeatures_C02_V01",
                 "/dataset/AIC2022/2/CLIPFeatures_C00_V02","/dataset/AIC2022/2/CLIPFeatures_C01_V02","/dataset/AIC2022/2/CLIPFeatures_C02_V02",
                 "/dataset/AIC2022/2/CLIPFeatures_C00_V03","/dataset/AIC2022/2/CLIPFeatures_C01_V03","/dataset/AIC2022/2/CLIPFeatures_C02_V03",
                 "/dataset/AIC2022/2/CLIPFeatures_C00_V04","/dataset/AIC2022/2/CLIPFeatures_C02_V04"
                 ],
# "L14_336": ["/workspace/sontda/prj/aci-2022/data/v00/l14_336/Features_C00_V00","/workspace/sontda/prj/aci-2022/data/v00/l14_336/Features_C01_V00","/workspace/sontda/prj/aci-2022/data/v00/l14_336/Features_C02_V00","/workspace/sontda/prj/aci-2022/data/v01/l14_336/Features_C00_V01","/workspace/sontda/prj/aci-2022/data/v01/l14_336/Features_C01_V01","/workspace/sontda/prj/aci-2022/data/v01/l14_336/Features_C02_V01"]}
"L14_336": ["/workspace/sontda/prj/aci-2022/data/v00/l14/Features_C00_V00"]}#,"/workspace/sontda/prj/aci-2022/data/v00/l14/Features_C01_V00","/workspace/sontda/prj/aci-2022/data/v00/l14/Features_C02_V00","/workspace/sontda/prj/aci-2022/data/v01/l14/Features_C00_V01","/workspace/sontda/prj/aci-2022/data/v01/l14/Features_C01_V01","/workspace/sontda/prj/aci-2022/data/v01/l14/Features_C02_V01"]}
FLAT_SIZE = {"B16": 512, "L14_336":768}
FAISS_PATH = "/dataset/AIC2022/faiss_index/"
# FAISS_PATH = "/dataset/AIC2022/faiss_indexxxx/"
KEYFRAME_FOLDER_PATH = "/dataset/AIC2022/"


### UTILS
class TextEmbedding(): # "ViT-B/16", "ViT-L/14@336px"
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name == "L14_336":
            self.model, _ = clip.load("ViT-L/14@336px", device=self.device)
        else:
            self.model, _ = clip.load("ViT-B/16", device=self.device)

    def __call__(self, text: str) -> np.ndarray:
        text_inputs = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            text_feature = self.model.encode_text(text_inputs)[0]
        print(text_inputs)
        return text_feature.detach().cpu().numpy()

def indexing_methods_faiss(model_name): # "ViT-B/16", "ViT-L/14@336px"
    global FAISS_PATH
    faiss_index_path = os.path.join(FAISS_PATH, "faiss_idx_{}.index".format(model_name))
    dict_path = os.path.join(FAISS_PATH, "idx_dict_{}.json".format(model_name))
    faiss_idx = faiss.IndexFlatL2(FLAT_SIZE[model_name])
    idx_list = []
    for folder_path in tqdm(CLIP_FEATURES_PATH[model_name]):
        # print(folder_path)
        for feat_npy in tqdm(os.listdir(folder_path)):
            # print(feat_npy)
            video_name = feat_npy.split('.')[0]
            feats_arr = np.load(os.path.join(folder_path, feat_npy))
            for idx, feat in enumerate(feats_arr):
            #Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes
                instance = (video_name, idx)
                idx_list.append(instance)
                faiss_idx.add(feat.reshape(1,-1).astype('float32'))
    idx_dict = dict(enumerate(idx_list))

    try:
        # save faiss dict
        with open(dict_path, 'w') as fw:
            json.dump(idx_dict, fw)
        faiss.write_index(faiss_idx, faiss_index_path)
        return idx_dict, faiss_idx
    except Exception as e:
        print(" !!! cannot save faiss index and dict !!! ")
        raise Exception(e)


def load_index_model(model_name): # "ViT-B/16", "ViT-L/14@336px"
    # global FAISS_PATH
    # faiss_index_path = os.path.join(FAISS_PATH, "faiss_idx_{}.index".format(model_name))
    # dict_path = os.path.join(FAISS_PATH, "idx_dict_{}.json".format(model_name))

    # if os.path.exists(faiss_index_path) and os.path.exists(dict_path):
    #     faiss_idx = faiss.read_index(faiss_index_path)
    #     with open(dict_path, 'r') as fr:
    #         idx_dict = json.load(fr)
    #     return idx_dict, faiss_idx
    # else:
    return indexing_methods_faiss(model_name)


### INIT VALUE
# model "ViT-B/16"
idx_dict_B16, faiss_idx_B16 = load_index_model("B16")
text_embedd_B16 = TextEmbedding("B16")

# model "ViT-L/14@336px"
idx_dict_L14_336, faiss_idx_L14_336 = load_index_model("L14_336")
text_embedd_L14_336 = TextEmbedding("L14_336")


app = flask.Flask("API for Similar")
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST', 'GET'])
def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    text, text2, query, model = "","","",""
    pil_img, image_url = None, None
    if request.method == "POST":
        text = request.json['text']
        text2 = request.json['text2']
        image_url = request.json['image_url']
        model_name = request.json['model']
    else:
        text = request.args.get('text')
        text2 = request.args.get('text2')
        # query = request.args.get('query')
        model_name = request.args.get('model')
        image_url = request.args.get('image_url')

    # get global params base on model
    if model_name == "B16":
        global idx_dict_B16, faiss_idx_B16, text_embedd_B16
        idx_dict, faiss_idx, text_embedd = idx_dict_B16, faiss_idx_B16, text_embedd_B16
    elif model_name == "L14_336":
        global idx_dict_L14_336, faiss_idx_L14_336, text_embedd_L14_336
        idx_dict, faiss_idx, text_embedd = idx_dict_L14_336, faiss_idx_L14_336, text_embedd_L14_336

    if image_url:
        image_url = image_url.lstrip()
        if (image_url[0:4]=="http"):
            response = requests.get(image_url)
            pil_img = Image.open(BytesIO(response.content))
        else:
            pil_img = Image.open(image_url)

    # preprocessing text 
    text_feat_arr = text_embedd(text).reshape(1,-1).astype('float32')
    # text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32')


    D, I = faiss_idx.search(text_feat_arr, k=200)

    search_results = []
    frames_id = []
    for instance in zip(I[0],D[0]):
        ins_id, distance = instance
        video_name, idx= idx_dict[ins_id]
        folder_vid = "2" if int(video_name[6]) >= 2 else video_name[6]
        frames_folder = KEYFRAME_FOLDER_PATH + folder_vid + "/KeyFrames"+ video_name[0:7] +'/'+ video_name
        keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
        video_name = video_name + '.mp4'
        result = {"video_name":str(video_name),
                  "keyframe_id": str(keyframe_id),
                  "score": str(distance)}
        search_results.append(result)
        frames_id.append(keyframe_id)

    if text2:
        text_feat_arr2 = text_embedd(text2).reshape(1,-1).astype('float32')
        D2, I2 = faiss_idx.search(text_feat_arr2, k=200)

        search_results2 = []
        for instance in zip(I2[0],D2[0]):
            ins_id, distance = instance
            video_name, idx= idx_dict[ins_id]
            folder_vid = "2" if int(video_name[6]) >= 2 else video_name[6]
            frames_folder = KEYFRAME_FOLDER_PATH + folder_vid + "/KeyFrames"+ video_name[0:7] +'/'+ video_name
            keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
            video_name = video_name + '.mp4'
            if keyframe_id in frames_id:
                result = {"video_name":str(video_name),
                        "keyframe_id": str(keyframe_id),
                        "score": str(distance)}
                search_results2.append(result)
        search_results = search_results2

    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True

    return response

# ##faiss
# def preprocessing_text(text):

#     global text_embedd
#     text_feat_arr = text_embedd(text)
#     text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32')
#     return text_feat_arr

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 9982, debug=False)
