import csv 
import os
import pandas as pd
path="/workspace/competitions/AI_Challenge_2022/results_v0"
output="/workspace/competitions/AI_Challenge_2022/results"
def read_csv(path_csv):
  df = pd.read_csv(path_csv, dtype=object)
  return df  

def convert_format(path, output_path):
    for file_name in os.listdir(path):
        file_path=os.path.join(path, file_name)
        out_path=os.path.join(output_path, file_name)
        data_df = read_csv(file_path)
        for i, keyf in enumerate(data_df.keyframe_id):
            num = 6-len(str(keyf))
            data_df.keyframe_id[i]='0'*num+keyf
            print(i,data_df.keyframe_id[i])
        data_df.to_csv(out_path,header=True,index=False)

convert_format(path, output)