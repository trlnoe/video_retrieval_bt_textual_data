#libraries 
import argparse

import os 
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image

import torch

from typing import List, Tuple

import json
import glob

def openJson(json_path): 
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data

def countingObject(data):
    final_object = []
    counting_str = ''
    for indx, score in enumerate(data['detection_scores']): 
        if float(score) >= 0.6: 
            final_object.append(data['detection_class_entities'][indx])
    for l in set(final_object): 
        counting_str=counting_str+l+' '+str(final_object.count(l))+' '
    return counting_str

def main(args):
    result = {}
    for folder in args.objects_folders: 
        for video_path in tqdm(glob.glob(folder)): 
            video_id = video_path.split('/')[-1]
            video_path+='/*'
            videoObject = {}
            for json_path in glob.glob(video_path):
                keyframe_id = json_path.split('/')[-1].replace('.json','.jpg')
                data = openJson(json_path)
                counting_str = countingObject(data)
                videoObject[keyframe_id] = counting_str
            result[video_id] = videoObject
    print(result['C02_V0375']['000768'])
    # with open("/workspace/competitions/AI_Challenge_2022/utils/countingObjects/objectCounting.json", "w") as outfile:
    #     json.dump(result, outfile)
  
#arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objects_folders", 
        default=["/dataset/AIC2022/0/ObjectsC00_V00/*","/dataset/AIC2022/0/ObjectsC01_V00/*","/dataset/AIC2022/0/ObjectsC02_V00/*",
                 "/dataset/AIC2022/1/ObjectsC00_V01/*","/dataset/AIC2022/1/ObjectsC01_V01/*","/dataset/AIC2022/1/ObjectsC02_V01/*",
                 "/dataset/AIC2022/2/ObjectsC00_V02/*","/dataset/AIC2022/2/ObjectsC01_V02/*","/dataset/AIC2022/2/ObjectsC02_V02/*",
                 "/dataset/AIC2022/2/ObjectsC00_V03/*","/dataset/AIC2022/2/ObjectsC01_V03/*","/dataset/AIC2022/2/ObjectsC02_V03/*",
                 "/dataset/AIC2022/2/ObjectsC00_V04/*","/dataset/AIC2022/2/ObjectsC02_V04/*",
                 ], 
        type=str, 
        help="CLIPFeature folder",
    )
    return parser.parse_args()

    
if __name__  == "__main__":
    args = get_parser()
    main(args)
    