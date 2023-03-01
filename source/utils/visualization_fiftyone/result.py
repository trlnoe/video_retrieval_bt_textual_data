#libraries 
import argparse

from calendar import c
import glob
import fiftyone as fo
import csv
import pandas as pd 
from tqdm import tqdm 
import json 


def keyframe_results(result_path): 
  df = pd.read_csv(result_path, dtype=object)
  result=df.values.tolist()
  return result


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


def main(args): 
    print("PORT: ", args.port)
    print("result_path: ", args.result_path)
    res = keyframe_results(args.result_path)
    #create samples for your data
    print('Generating a dataset...')
    samples = []
    for i in tqdm(res): 
        filepath = f'{args.keyframe_folder_path}{i[0][6]}/KeyFrames{i[0][0:7]}/{i[0].split(".")[0]}/{str(i[1])}.jpg'
        sample = fo.Sample(filepath=filepath)
        samples.append(sample)
 
    # fo.load_dataset("AIC2022-Result").delete()
    fo.config.default_app_port = args.port
    print(fo.config.default_app_port)
    dataset = fo.Dataset()
    dataset.add_samples(samples)
    print("Done")
    # #draw bounding box object 
    # print('Generating a object bounding box...')
    # dataset = object_bounding_box(dataset)
    # print('Done')
    #web browser
    session = fo.launch_app(dataset)
    session.wait()

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
        "--result_path", 
        default="/workspace/competitions/AI_Challenge_2022/results/query-2.csv", 
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
        "--dataset_name", 
        default="AIC2022-Result", 
        type=str, 
        help="Dataset name",
    )
    parser.add_argument(
        "--port", 
        default=5151, 
        type=int, 
        help="Dataset name",
    )
    return parser.parse_args()

if __name__ == "__main__":
  args = get_parser() 
  main(args)
