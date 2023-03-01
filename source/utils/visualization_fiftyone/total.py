#libraries 
import argparse

from calendar import c
import glob
import fiftyone as fo
import csv
from tqdm import tqdm 
import json 
import numpy as np 

def generate_dataset(dataset_path): 
  #generate dataset
  dataset = fo.Dataset.from_images_dir(dataset_path, name=None, tags=None, recursive=True)
  return dataset

def video_frameid_info(dataset): 
  #add video and frame info 
  for sample in tqdm(dataset):
    _, sample['video'], sample['frameid'] = sample['filepath'].rsplit('/', 2)
    sample.save()
  #check dataset info
  print(dataset.first())
  return dataset

def object_bounding_box(dataset): 
  for sample in tqdm(dataset):
    #open matching json file 
    object_path = '{}/{}.json'.format(args.object_path,sample['filepath'][-20:-4])
    with open(object_path) as jsonfile:
        det_data = json.load(jsonfile)
        
    detections = []
    for cls, box, score in zip(det_data['detection_class_entities'], det_data['detection_boxes'], det_data['detection_scores']):
        # Convert to [top-left-x, top-left-y, width, height]
        boxf = [float(box[1]), float(box[0]), float(box[3]) - float(box[1]), float(box[2]) - float(box[0])]
        scoref = float(score)
        
        # Only add objects with confidence > 0.4
        if scoref > 0.4:
            detections.append(
                fo.Detection(
                    label=cls,
                    bounding_box= boxf,
                    confidence=float(score)
                )
            )
    sample["object_faster_rcnn"] = fo.Detections(detections=detections)
    sample.save()
  #check dataset info
  print(dataset.first())
  return dataset 

def keyframe_results(result_path): 
  with open(result_path, newline='') as f:
    reader = csv.reader(f)
    result = list(reader)
    f.close()
  return result[1:]

def main(args): 
  #create dataset
  print('Generating a dataset...')
  dataset = generate_dataset(args.dataset_path)
  print('Done')
  
  #insert video and frameid 
  print('Generating a video and frameid...')
  dataset = video_frameid_info(dataset)
  print('Done')
  
  #draw bounding box object 
  print('Generating a object bounding box...')
  dataset = object_bounding_box(dataset)
  print('Done')
  
  
  #open webrower
  session = fo.launch_app(dataset)
  session.wait()

#arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", 
        default="/dataset/AIC2022/0/KeyFramesC00_V00", 
        type=str, 
        help="Keyframe folder",
    )
    parser.add_argument(
        "--object_path", 
        default="/dataset/AIC2022/0/ObjectsC00_V00", 
        type=str, 
        help="Keyframe folder",
    )
    parser.add_argument(
        "--clip_features_path", 
        default="/dataset/AIC2022/0/CLIPFeatures_C00_V00", 
        type=str, 
        help="CLIPFeature folder",
    )
    parser.add_argument(
        "--result_path", 
        default="/workspace/competitions/AI_Challenge_2022/results/result_baseline.csv", 
        type=str, 
        help="Keyframe folder",
    )
    return parser.parse_args()

if __name__ == "__main__":
  args = get_parser() 
  main(args)
