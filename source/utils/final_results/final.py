import pandas as pd 
import glob 

def final_result(in_path, out_path, key_path):

    #in_path : địa chỉ file query
    #out_path: địa chỉ file xuất
    #key_path: địa chỉ chứa folder keyframe
    for path in glob.glob(in_path): 
        #Get query info
        query = pd.read_csv(path,names=["video_name","keyframe_id","score"])
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
        out_ = out_path+path.split('/')[-1]
        print(out_)
        df.to_csv(out_,index=False, header=False)
    
final_result('/workspace/competitions/AI_Challenge_2022/results/*', '/workspace/competitions/AI_Challenge_2022/submission/','/dataset/AIC2022/Keyframe_P/')