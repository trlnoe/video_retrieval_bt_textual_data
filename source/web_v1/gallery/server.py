# import numpy as np
# from PIL import Image
# from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
from pathlib import Path
import os
app = Flask(__name__)

basepath = "/dataset/AIC2022/"

path_a = "0"
path_b = "KeyFramesC00_V00"
path_c = "C00_V0000"

path_a_lst = ["0", "1"]
path_b_lst = ["KeyFramesC00_V0", "KeyFramesC01_V0", "KeyFramesC02_V0"]
path_c_lst = [str(i).zfill(2) for i in range(0, 100)]  # 00 --> 99


@app.route('/img/<path:filename>')
def download_file(filename):
    # print(os.path.join('/', 'dataset', 'AIC2022', path_a, path_b, path_c, filename))
    return send_from_directory(os.path.join('/', 'dataset', 'AIC2022', path_a, path_b, path_c), filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global path_a, path_b, path_c
        path_a = request.form['path_a_option']
        path_b = request.form['path_b_option']
        path_c = request.form['path_c_option']

        path_b = path_b + path_a
        path_c = path_b[-7:] + path_c

        global basepath
        files = [f for f in os.listdir(
            os.path.join(basepath, path_a, path_b, path_c))]
        files.sort()
        scores = [('zero', f) for f in files]
        return render_template('index.html', path_a=path_a_lst, path_b=path_b_lst, path_c=path_c_lst, scores=scores, original_path=os.path.join(path_a, path_b, path_c), video_path=path_c)

    else:
        basepath = "/dataset/AIC2022/"
        mypath = "/dataset/AIC2022/0/KeyFramesC00_V00/C00_V0000"
        files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
        files.sort()
        scores = [('zero', f) for f in files]

        return render_template('index.html', scores=scores, path_a=path_a_lst, path_b=path_b_lst, path_c=path_c_lst, original_path="0/KeyFramesC00_V00/C00_V0000", video_path=path_c)


if __name__ == "__main__":
    app.run("0.0.0.0", port=8081)
