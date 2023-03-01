from inspect import ArgSpec
import pandas as pd
import argparse

def getName(data_path, date, month, year):
  df = pd.read_csv(data_path)
  res = []
  for i in range(0,df.publish_date.shape[0]):
    cur = df.publish_date[i]
    curDate = ""
    curMonth = ""
    curYear = ""
    j = 0
    while (cur[j]!='/'):
      curDate+=cur[j]
      j+=1
    curDate = int(curDate)
    j+=1
    while (cur[j]!='/'):
      curMonth+=cur[j]
      j+=1
    curMonth = int(curMonth)
    j+=1
    while (j<len(cur)):
      curYear+=cur[j]
      j+=1
    curYear = int(curYear)
    #print(curDate, date)
    if (int(date)==curDate and int(month)==curMonth):
      res.append([df.video[i], df.publish_date[i]])
  print(res)
  
def main(args):
  getName('/dataset/AIC2022/metadata.csv',args.date, args.month, args.year)

#arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", 
        default=1, 
        type=int, 
        help="date",
    )
    parser.add_argument(
        "--month", 
        default=1, 
        type=int, 
        help="month",
    )
    parser.add_argument(
        "--year", 
        default=2022, 
        type=int, 
        help="year",
    )
    return parser.parse_args()

    
if __name__  == "__main__":
    args = get_parser()
    main(args)