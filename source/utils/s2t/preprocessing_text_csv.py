import csv 
import tqdm

def processing(folder_path): 
    for file in tqdm(folder_path): 
        with open(file) as f:
            reader = csv.reader(f)
            data = list(reader)
            data = [line for line in data if len(line) < 30]
            write = csv.writer(f)
            write.writerows(data)
        f.close()


processing('/workspace/competitions/AI_Challenge_2022/source/whisper/csv_S2T')