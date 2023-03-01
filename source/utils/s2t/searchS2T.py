#libraries 
import argparse

import os 
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image

import torch
import clip
import faiss

from typing import List, Tuple

import csv 
import glob
import unidecode
 

#funtions 
##text_mbedding 
##faiss 
class TextEmbedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/16", device=self.device)

    def __call__(self, text: str) -> np.ndarray:
        text_inputs = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(text_inputs)[0]
        return text_feature.detach().cpu().numpy()
    
def indexing_methods_faiss():
    faiss_db = faiss.IndexFlatL2(512)
    db = []
    for file in glob.glob(args.folder_path):
        # print(file)
        with open(file, 'r') as f:
            data = f.read().split('\n')
        for index, text in tqdm(enumerate(data)):
            # print(feat_npy)
            video_name = file.split('/')[-1].split('.')[0] 
            instance = (video_name, index)
            db.append(instance)
            text_embedd = TextEmbedding()
            text_feat_arr = text_embedd(text.replace('\n',''))
            text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32')
            faiss_db.add(text_feat_arr)
    db = dict(enumerate(db))
    return db, faiss_db

#main function 
def main(args):
    db, faiss_db = indexing_methods_faiss()
    text = unidecode.unidecode(args.text)
    text_embedd = TextEmbedding()
    text_feat_arr = text_embedd(text)
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32')
    D, I = faiss_db.search(text_feat_arr, k=100)
    search_result = []
    
    for instance in tqdm(zip(I[0],D[0])):
        ins_id, distance = instance
        video_name, idx= db[ins_id]
        video_name = video_name 
        search_result.append({"video_name":video_name,
                                "line": idx})
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
        default="Lễ kỷ niệm 132 ngày sinh chủ tịch hồ chí minh", 
        type=str, 
        help="Text Search",
    )
    return parser.parse_args()

    
if __name__  == "__main__":
    args = get_parser()
    main(args)
    


    
    
