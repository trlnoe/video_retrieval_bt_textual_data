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
#funtions 
##text_mbedding 
class TextEmbedding():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/16", device=self.device)

    def __call__(self, text: str) -> np.ndarray:
        text_inputs = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_feature = self.model.encode_text(text_inputs)[0]
        # print(text_inputs)
        return text_feature.detach().cpu().numpy()



##faiss 
def indexing_methods_faiss():
    faiss_db = faiss.IndexFlatL2(512)
    db = []
    for folder_path in tqdm(args.clip_features_path):
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
##indexing 
def indexing_methods() -> List[Tuple[str, int, np.ndarray],]:
    db = []
    #Duyệt tuần tự và đọc các features vector từ file .npy
    for feat_npy in tqdm(os.listdir(args.clip_features_path)):
        video_name = feat_npy.split('.')[0]
        feats_arr = np.load(os.path.join(args.clip_features_path, feat_npy))
        for idx, feat in enumerate(feats_arr):
        #Lưu mỗi records với 3 trường thông tin là video_name, keyframe_id, feature_of_keyframes
            instance = (video_name, idx, feat)
            db.append(instance)
    return db

#search_engine 
def search_engine(query_arr: np.array, 
                  db: list, 
                  topk:int=10, 
                  measure_method: str="dot_product") -> List[dict,]:
  
    '''Duyệt tuyến tính và tính độ tương đồng giữa 2 vector'''
    measure = []
    for ins_id, instance in enumerate(db):
        video_name, idx, feat_arr = instance

        if measure_method=="dot_product":
            distance = query_arr @ feat_arr.T
        elif measure_method=="l1_norm":
            distance = -1 * np.mean([abs(q - t) for q, t in zip(query_arr, feat_arr)])
        measure.append((ins_id, distance))
  
    '''Sắp xếp kết quả'''
    measure = sorted(measure, key=lambda x:x[-1], reverse=True)
  
    '''Trả về top K kết quả'''
    search_result = []
    for instance in measure[:topk]:
        ins_id, distance = instance
        video_name, idx, _ = db[ins_id]
        keyframe_id = sorted(os.listdir(os.path.join(args.image_keyframe_path, video_name)))[idx].split('.')[0]
        video_name = video_name + '.mp4'
        search_result.append({"video_name":video_name,
                                "keyframe_id": keyframe_id,
                                "score": distance})
    return search_result
#merge right keyframe
def final_result(in_path, out_path, key_path):

    #in_path : địa chỉ file query
    #out_path: địa chỉ file xuất
    #key_path: địa chỉ chứa folder keyframe

    #Get query info
    query = pd.read_csv(in_path,names=["video_name","keyframe_id","score"])
    querySize = query.shape[0]

    #Process
    res = []
    for i in range(1,querySize):

        #Create file name & get its info
        s=query.video_name[i][0:9:1]
        cur_path = key_path + s + ".csv"
        curVideo = pd.read_csv(cur_path,names=["oldframe","newframe"])
        curVideo.sort_values(by=["oldframe", "newframe"], inplace=True)
        #sort and binary search to quick search a frame value
        left=0
        right=curVideo.shape[0]-1
        while (left<=right):
            mid=int((left+right)/2)
            a = int(query.keyframe_id[i])
            b = int(curVideo.oldframe[mid][0:6:1])
            if (a == b): #found
                res.append([query.video_name[i],curVideo.newframe[mid]])
                break
            if (a>b):
                left=mid+1
            else:
                right=mid-1

    #turn list into data frame for exporting
    df = pd.DataFrame(res, columns=["video_name", "keyframe_id"])
    df.to_csv(out_path,index=False, header=False)
 
def query_reading(result_path): 
  df = pd.read_csv(result_path, dtype=object, header=None)
  query_list=df.values.tolist()
  return query_list   
#main function 
def main(args):
    db, faiss_db = indexing_methods_faiss()
    text = args.text
    text_embedd = TextEmbedding()
    text_feat_arr = text_embedd(text)
    text_feat_arr = text_feat_arr.reshape(1,-1).astype('float32')
    D, I = faiss_db.search(text_feat_arr, k=100)
    
    search_result = []
    
    for instance in zip(I[0],D[0]):
        ins_id, distance = instance
        video_name, idx= db[ins_id]
        frames_folder = args.keyframe_folder_path + video_name[6] + "/KeyFrames"+ video_name[0:7] +'/'+ video_name
        keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
        video_name = video_name + '.mp4'
        search_result.append({"video_name":video_name,
                                "keyframe_id": keyframe_id,
                                "score": distance})
    ##saving csv 
    ###results
    # keys = search_result[0].keys()
    # with open('/workspace/competitions/AI_Challenge_2022/results/result_faiss_baseline.csv', 'w', newline='') as output_file:
    #     dict_writer = csv.DictWriter(output_file, keys)
    #     dict_writer.writeheader()
        # dict_writer.writerows(search_result)
    
    ###submission 
    video_id = list(item['video_name'] for item in search_result) 
    frame_id = list(item['keyframe_id'] for item in search_result) 
    score = list(item["score"] for item in search_result)
    rows = zip(video_id,frame_id,score)
    
    result_path_saving = args.result_csv + "query-" + str(args.query) +'.csv'
    submission_path_saving = args.submission_csv +  "query-" + str(args.query) +'.csv'
    with open(result_path_saving, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["video_name", "keyframe_id", "score"])
        for row in rows:
            writer.writerow(row)
    
    ###final: 
    final_result(result_path_saving, submission_path_saving, args.keyframe_position)
        
    
  
#arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyframe_folder_path", 
        default="/dataset/AIC2022/", 
        type=str, 
        help="Keyframe folder",
    )
    parser.add_argument(
        "--clip_features_path", 
        default=["/dataset/AIC2022/0/CLIPFeatures_C00_V00","/dataset/AIC2022/0/CLIPFeatures_C01_V00","/dataset/AIC2022/0/CLIPFeatures_C02_V00",
                 "/dataset/AIC2022/1/CLIPFeatures_C00_V01","/dataset/AIC2022/1/CLIPFeatures_C01_V01","/dataset/AIC2022/1/CLIPFeatures_C02_V01"], 
        type=str, 
        help="CLIPFeature folder",
    )
    parser.add_argument(
        "--topk", 
        default=100, 
        type=int, 
        help="Top Keyframe quantity",
    )
    parser.add_argument(
        "--text", 
        default="hills and mountains, many balloons, a Seagame 31 symbol on large balloon", 
        type=str, 
        help="Text Query",
    )
    parser.add_argument(
        "--query", 
        default=26, 
        type=int, 
        help="Text Query",
    )
    parser.add_argument(
        "--result_csv", 
        default="/workspace/competitions/AI_Challenge_2022/results/", 
        type=str, 
        help="Results file path",
    )
    parser.add_argument(
        "--submission_csv", 
        default="/workspace/competitions/AI_Challenge_2022/submission/", 
        type=str, 
        help="Submission file path",
    )
    parser.add_argument(
        "--keyframe_position", 
        default="/dataset/AIC2022/Keyframe_P/", 
        type=str, 
        help="Keyframe position path",
    )
    parser.add_argument(
        "--query_path", 
        default="/dataset/AIC2022/query/query-pack-0.csv", 
        type=str, 
        help="Query path",
    )
    return parser.parse_args()

    
if __name__  == "__main__":
    args = get_parser()
    main(args)
    


    
    
