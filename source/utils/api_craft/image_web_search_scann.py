import flask
from flask import request
import os
import numpy as np
from tqdm import tqdm
from time import  time
import torch
import sys
import json
import tensorflow as tf
sys.path.append("/workspace/sontda/prj/CLIP")
print("sys path: ", sys.path) 

import clip
import scann

### STATIC VALUE
CLIP_FEATURES_PATH = {"B16": ["/dataset/AIC2022/0/CLIPFeatures_C00_V00","/dataset/AIC2022/0/CLIPFeatures_C01_V00","/dataset/AIC2022/0/CLIPFeatures_C02_V00","/dataset/AIC2022/1/CLIPFeatures_C00_V01","/dataset/AIC2022/1/CLIPFeatures_C01_V01","/dataset/AIC2022/1/CLIPFeatures_C02_V01"],
"L14_336": ["/workspace/sontda/prj/aci-2022/data/v00/l14_336/Features_C00_V00","/workspace/sontda/prj/aci-2022/data/v00/l14_336/Features_C01_V00","/workspace/sontda/prj/aci-2022/data/v00/l14_336/Features_C02_V00","/workspace/sontda/prj/aci-2022/data/v01/l14_336/Features_C00_V01","/workspace/sontda/prj/aci-2022/data/v01/l14_336/Features_C01_V01","/workspace/sontda/prj/aci-2022/data/v01/l14_336/Features_C02_V01"]}
FLAT_SIZE = {"B16": 512, "L14_336":768}
SCANN_PATH = "/dataset/AIC2022/scann_index/"
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

def indexing_methods_scann(model_name): # "ViT-B/16", "ViT-L/14@336px"
    global SCANN_PATH
    scann_index_path = os.path.join(SCANN_PATH, "scann_seacher_{}".format(model_name))
    if not os.path.exists(scann_index_path):
        os.makedirs(scann_index_path)
    dict_path = os.path.join(SCANN_PATH, "idx_dict_{}.json".format(model_name))
    
    idx_list = []
    all_npy = np.empty((0,FLAT_SIZE[model_name]))
    for folder_path in tqdm(CLIP_FEATURES_PATH[model_name]):
        # print(folder_path)
        for feat_npy in tqdm(os.listdir(folder_path)):
            # print(feat_npy)
            video_name = feat_npy.split('.')[0]
            feats_arr = np.load(os.path.join(folder_path, feat_npy))
            all_npy = np.append(all_npy, feats_arr, axis=0)
            for idx in range(len(feats_arr)):
                instance = (video_name, idx)
                idx_list.append(instance)

    idx_dict = dict(enumerate(idx_list))

    ### create scann
    k = int(np.sqrt(all_npy.shape[0]))
    scann_searcher = scann.scann_ops_pybind.builder(all_npy, 10, "squared_l2").tree(
        num_leaves=k, num_leaves_to_search=int(k/5), training_sample_size=2500).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()

    # return idx_dict, scann_searcher
    try:
        # save scann dict
        with open(dict_path, 'w') as fw:
            json.dump(idx_dict, fw)
        # save scann index
        scann_searcher.serialize(scann_index_path)
        return idx_dict, scann_searcher
    except Exception as e:
        print(" !!! cannot save scann index and dict !!! ")
        raise Exception(e)


def load_index_model(model_name): # "ViT-B/16", "ViT-L/14@336px"
    global SCANN_PATH
    scann_index_path = os.path.join(SCANN_PATH, "scann_seacher_{}".format(model_name))
    dict_path = os.path.join(SCANN_PATH, "idx_dict_{}.json".format(model_name))

    if os.path.exists(scann_index_path) and os.path.exists(dict_path):
        scann_searcher = scann.scann_ops_pybind.load_searcher(scann_index_path)
        with open(dict_path, 'r') as fr:
            idx_dict = json.load(fr)
        return idx_dict, scann_searcher
    else:
        return indexing_methods_scann(model_name)


def get_search_value(idx_dict, searcher, text_embedd, text_search):
    # preprocessing text 
    text_feat_arr = text_embedd(text_search)
    text_feat_arr = text_feat_arr.astype('float16')

    I, D = searcher.search(text_feat_arr, final_num_neighbors=200)
    search_results = []
    for instance in zip(I,D):
        ins_id, distance = instance
        video_name, idx= idx_dict[ins_id]
        frames_folder = KEYFRAME_FOLDER_PATH + video_name[6] + "/KeyFrames"+ video_name[0:7] +'/'+ video_name
        keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
        video_name = video_name + '.mp4'
        result = {"video_name":str(video_name),
                  "keyframe_id": str(keyframe_id),
                  "score": str(distance)}
        print("result: ", result)
        search_results.append(result)
    return search_results

### INIT VALUE
# model "ViT-B/16"
idx_dict_B16, scann_seacher_B16 = load_index_model("B16")
text_embedd_B16 = TextEmbedding("B16")

# model "ViT-L/14@336px"
idx_dict_L14_336, scann_seacher_L14_336 = load_index_model("L14_336")
text_embedd_L14_336 = TextEmbedding("L14_336")


app = flask.Flask("API for Similar")
app.config["DEBUG"] = True

@app.route('/predict', methods=['POST', 'GET'])
def updateCurrentCode():
    global KEYFRAME_FOLDER_PATH
    text =""
    query = ""
    model = ""
    if request.method == "POST":
        text = request.json['text']
        # query = request.json['query']
        model_name = request.json['model']
    else:
        text = request.args.get('text')
        # query = request.args.get('query')
        model_name = request.args.get('model')

    # get global params base on model
    search_results = []
    if model_name == "B16":
        global idx_dict_B16, scann_seacher_B16, text_embedd_B16
        search_results = get_search_value(idx_dict_B16, scann_seacher_B16, text_embedd_B16, text)
    elif model_name == "L14_336":
        global idx_dict_L14_336, scann_seacher_L14_336, text_embedd_L14_336
        search_results = get_search_value(idx_dict_L14_336, scann_seacher_L14_336, text_embedd_L14_336, text)

    response = flask.jsonify(search_results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True

    return response

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port= 8019, debug=False)
    from waitress import serve
    serve(app, host='0.0.0.0', port=8019)
