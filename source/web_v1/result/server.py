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
app = Flask(__name__)


basepath = "/dataset/AIC2022/"
true_frame_path = "/dataset/AIC2022/Keyframe_P_JSON"
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
    return send_from_directory(directory="/" + directory, path=video_name)


@app.route('/video/<path:filename>/<path:keyframe>')
def video(filename, keyframe):
    filename = filename + '/' + keyframe
    splitted = filename.split("/dataset/")
    video_path = splitted[0]
    fps = get_fps(video_path)
    true_key_frame = mapping_true_frame(keyframe, video_path)
    keyframe_path = math.floor(true_key_frame / fps)
    print("keyframe_path: ", keyframe_path)
    return render_template('video.html', source=video_path, keyframe=keyframe_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    files = os.listdir('/workspace/competitions/AI_Challenge_2022/results')
    if request.method == 'POST':
        query_filename = request.form['query']
        query_filename = os.path.join(
            '/workspace/competitions/AI_Challenge_2022/results', query_filename)

        results = []
        lines = []
        with open(query_filename, 'r') as f:
            lines = f.read().splitlines()[1:]

        for i in range(0, len(lines)):
            splitted = lines[i].split(",")
            folder_a = splitted[0][0:9]
            video_path = os.path.join(
                "dataset", "AIC2022", folder_a[-3], "Video" + folder_a[0:-2], folder_a + ".mp4")
            path = os.path.join(
                "dataset", "AIC2022", folder_a[-3], "KeyFrames" + folder_a[0:-2], folder_a, splitted[1] + ".jpg")
            results.append((
                i, #so dong
                folder_a,#folder video 
                splitted[1],
                path,
                splitted[2],
                video_path
            ))
        return render_template('detail.html', files=files, results=results)
    else:
        return render_template('detail.html', files=files, results=[])


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


if __name__ == "__main__":
    app.run("0.0.0.0", port=8083)
