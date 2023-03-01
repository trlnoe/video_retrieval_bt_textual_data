# import numpy as np
# from PIL import Image
# from feature_extractor import FeatureExtractor
import cv2
import math
import requests
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
from pathlib import Path
import os
import json 
import pickle
app = Flask(__name__)

true_frame_path = "/dataset/AIC2022/Keyframe_P_JSON"
basepath = "/dataset/AIC2022/"

path_a = "0"
path_b = "KeyFramesC00_V00"
path_c = "C00_V0000"

path_a_lst = ["0", "1"]
path_b_lst = ["KeyFramesC00_V0", "KeyFramesC01_V0", "KeyFramesC02_V0"]
path_c_lst = [str(i).zfill(2) for i in range(0, 100)]  # 00 --> 99


@app.route('/img/<path:filename>')
def download_file(filename):
    directory = "/".join(filename.split("/")[:-1])
    video_name = filename.split("/")[-1]
    directory = directory.replace("AIC2022/3/", "AIC2022/2/")
    directory = directory.replace("AIC2022/4/", "AIC2022/2/")
    return send_from_directory(directory="/" + directory, path=video_name)

@app.route('/video/<path:filename>/<path:keyframe>')
def video(filename, keyframe):
    filename = filename + '/' + keyframe
    splitted = filename.split("/dataset/")
    video_path = splitted[0]
    fps = get_fps(video_path)
    true_key_frame = mapping_true_frame(keyframe, video_path)
    keyframe_path = math.floor(true_key_frame / fps)
    mi = str(keyframe_path//60)
    if len(mi)==1: 
        mi="0"+mi
    se = str(int(keyframe_path%60))
    if len(se)==1: 
        se="0"+se
    right_id = video_path.split("/")[-1] + ", " + mi +":"+se
    return render_template('video.html', source=video_path, keyframe=keyframe_path, id = right_id)


def get_fps(video_path):
    video = cv2.VideoCapture("/"+video_path)
    fps = round(video.get(cv2.CAP_PROP_FPS))
    return fps


def mapping_true_frame(keyframe, video_path):
    video_name = video_path[:-4].split("/")[-1]
    keyframe_img = keyframe.split("/")[-1]
    json_path = os.path.join(true_frame_path, video_name+".json")
    with open(json_path, 'r') as openfile:
        # Reading from json file
        json_data = json.load(openfile)
    true_key_frame = int(json_data[keyframe_img])
    return true_key_frame

@app.route('/', methods=['GET', 'POST'])
def index():
    global org2id_dict
    if request.method == 'POST':
        text = request.form['query']
        image = request.form['fname']
        objs = request.form['objs']

        if text != "": 
            url_text = "http://172.17.0.4:8015/predict?text={}&model=B16".format(text)
            result = requests.get(url_text).json()
            lst_video_name_text = [(res['video_name'], res['keyframe_id'])
                            for res in result]
        else: 
            lst_video_name_text = []

        if image != "": 
            url_text = "http://172.17.0.4:8019/predict?image_url={}".format(image)
            result = requests.get(url_text).json()
            lst_video_name_img = [(res['video_name'], res['keyframe_id'])
                            for res in result]
        else: 
            lst_video_name_img = []
            
        if objs != "": 
            url_text = "http://172.17.0.5:5000/predict?object_str={}".format(objs)
            result = requests.get(url_text).json()
            lst_video_name_obj = [(res['video_name'], res['keyframe_id'])
                            for res in result]
        else: 
            lst_video_name_obj = []
        
        lst_video_name = lst_video_name_text + lst_video_name_img + lst_video_name_obj
        
        files = []

        for _, video_name in enumerate(lst_video_name):
            folder_a = video_name[0][0:9]
            
            if ".jpg" not in video_name[1]: 
                frame = video_name[1] + ".jpg"
            else: 
                frame = video_name[1]
            path = os.path.join("dataset", "AIC2022", folder_a[-3], "KeyFrames" + folder_a[0:-2], folder_a, frame)
            path = path.replace("AIC2022/3/", "AIC2022/2/")
            path = path.replace("AIC2022/4/", "AIC2022/2/")
            # right_id = folder_a + ", " + org2id_dict[folder_a][path.split("/")[-1]]
            video_path = os.path.join("dataset", "AIC2022", folder_a[-3], "Video" + folder_a[0:-2], video_name[0])
            video_path = video_path.replace("AIC2022/3/", "AIC2022/2/")
            video_path = video_path.replace("AIC2022/4/", "AIC2022/2/")
            
            if ".mp4" not in video_path: 
                video_path+=".mp4"
            
            v_name = str(video_name[0][0:15])
            if ".mp4" not in v_name: 
                v_name+=".mp4"   
            fps = int(video_fps[v_name])
            keyframe_path = math.floor(int(org2id_dict[folder_a][path.split("/")[-1]])/ fps)
            mi = str(keyframe_path//60)
            if len(mi)==1: 
                mi="0"+mi
            se = str(int(keyframe_path%60))
            if len(se)==1: 
                se="0"+se
            right_id = folder_a + ", " + mi +":"+se
            
            #true_key_frame = mapping_true_frame(path, video_path)
            #keyframe_path = math.floor(true_key_frame / fps)
            
            files.append((_, path, right_id, video_path, keyframe_path))
        return render_template('index.html', files=files, query=text, image=image, objs=objs, count=str(len(files)) +' files found.')
    else:
        mypath = "/dataset/AIC2022/0/KeyFramesC00_V00/C00_V0000"
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        files.sort()    
        scores = [('zero', f) for f in files]

        return render_template('index.html', scores=scores, path_a=path_a_lst, path_b=path_b_lst, path_c=path_c_lst, original_path="0/KeyFramesC00_V00/C00_V0000", video_path=path_c)
    

if __name__ == "__main__":
    org2id_dict_path = "/dataset/AIC2022/map_org2true_id.json"
    with open(org2id_dict_path) as json_file:
        org2id_dict = json.load(json_file)
    
    video_fps_path = '/dataset/AIC2022/video_fps.pickle'
    with open(video_fps_path, 'rb') as handle:
        video_fps = pickle.load(handle)
    app.run("0.0.0.0", port=8111)
