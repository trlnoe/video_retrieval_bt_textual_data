#libraries 
import argparse

import os 
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image

import torch
import clip

from typing import List, Tuple

import csv 
import glob
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
#funtions 
##text_mbedding 
##faiss 

#main function 
def main(args):
    search_text = args.text
    search_result = []
    for file in glob.glob(args.folder_path):
        # print(file)
        with open(file, 'r') as f:
            data = f.read().split('\n')
        for index, text in tqdm(enumerate(data)):
            # print(feat_npy)
            video_name = file.split('/')[-1].split('.')[0] 
            instance = (video_name, index)
            
            if similar(text,search_text) >= 0.5:
                search_result.append(instance)
    print(search_result)


#arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path", 
        default="/workspace/competitions/AI_Challenge_2022/source/whisper/AIC_S2T_text/*", 
        type=str, 
        help="csv S2T folder",
    )
    parser.add_argument(
        "--text", 
        default="vẫn còn những vấn đề cần tiếp tục tháo gỡ", 
        type=str, 
        help="Text Search",
    )
    return parser.parse_args()

    
if __name__  == "__main__":
    args = get_parser()
    main(args)
    


    
    
