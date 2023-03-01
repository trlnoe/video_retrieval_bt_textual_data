# import numpy as np
# from PIL import Image
# from feature_extractor import FeatureExtractor
import requests
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
from pathlib import Path
import os
import json 

app = Flask(__name__)


basepath = "/dataset/AIC2022/"
org2id_dict = {}
# path_a = "0"
# path_b = "KeyFramesC00_V00"
# path_c = "C00_V0000"

# path_a_lst = ["0","1","2"]
# path_b_lst = ["KeyFramesC00_V0", "KeyFramesC01_V0", "KeyFramesC02_V0"]
# path_c_lst = [str(i).zfill(2) for i in range(0, 100)]  # 00 --> 99


@app.route('/img/<path:filename>')
def download_file(filename):
    directory = "/".join(filename.split("/")[:-1])
    video_name = filename.split("/")[-1]
    return send_from_directory(directory="/" + directory, path=video_name)


@app.route('/', methods=['GET', 'POST'])
def index():
    global org2id_dict
    if request.method == 'POST':
        text_query = request.form['text_query']
        image_query = request.form['image_query']
        text_query2 = request.form['text_query2']
        objects_query = request.form['objetcs_query']
        
        include_text = request.form['text']
        include_image = request.form['image']
        include_text2= request.form['text_2']
        include_objects = request.form['objects']
        
        if text_query == '': 
            include_text = "OFF"
        else: 
            include_text = "ON"
        
        if image_query == '': 
            include_image = "OFF"
        else: 
            include_image = "ON"
            
        if objects_query == '': 
            include_objects = "OFF"
        else: 
            include_objects = "ON"
        
        # call api 
        if include_text=="ON": 
            url_text = "http://192.168.1.252:9982/predict?text={}&text2={}&model=B16".format(text_query,text_query2)
            text_result  = requests.get(url_text).json()
            # print(text_result)
            lst_video_name_by_text = [(res['video_name'], res['keyframe_id'])
                          for res in text_result]
        else: 
            lst_video_name_by_text=[]
        
        
        if include_image=="ON": 
            url_image = "http://192.168.1.252:8019/predict?image_url={}".format(image_query)
            image_result  = requests.get(url_image).json()
            lst_video_name_by_image = [(res['video_name'], res['keyframe_id'])
                            for res in image_result]
        else: 
            lst_video_name_by_image=[]
        
        if include_objects=="ON": 
            objects_text = "http://192.168.1.252:8027/predict?object_str={}".format(objects_query)
            objects_result  = requests.get(objects_text).json()
            lst_video_name_by_objects = [(res['video_name'], res['keyframe_id'])
                            for res in objects_result]
        else: 
            lst_video_name_by_objects=[]
        
        #merge result 
        total = set(lst_video_name_by_text + lst_video_name_by_image + lst_video_name_by_objects)
        quantity=len(total)

        files = []
        for video_name in total:
            folder_a = video_name[0][0:9]
            if folder_a[-3]=='3' or folder_a[-3]=='4': 
                folder='2'
            else: 
                folder = folder_a[-3]
            frame = ' '
            if ".jpg" not in str(video_name[1]): 
                frame = str(video_name[1]) + '.jpg'
            else: 
                frame = str(video_name[1])
            path = os.path.join(
                "dataset", "AIC2022", folder, "KeyFrames" + folder_a[0:-2], folder_a, frame)
            right_id = folder_a + ", " + org2id_dict[folder_a][frame]
            files.append((path,right_id))
        previous = text_query+"\n"+text_query2+"\n"+image_query+"\n"+"\n"+objects_query
        return render_template('index.html',files=files, query=previous,quantity=quantity)
    else:
        mypath = "/dataset/AIC2022/0/KeyFramesC00_V00/C00_V0000"
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        files.sort()
        scores = [('zero', f) for f in files]

        return render_template('index.html')


if __name__ == "__main__":
    org2id_dict_path = "/dataset/AIC2022/map_org2true_id.json"
    with  open(org2id_dict_path) as json_file:
        org2id_dict = json.load(json_file)
    app.run("0.0.0.0", port=8023)
