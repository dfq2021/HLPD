import json
import argparse
import os

def count_words(json_file, nums):
    with open(json_file,"r",encoding="utf-8") as f:
        data = json.load(f)
    cnt = 0
    for _, val in data.items():
        for sentence in val[:nums]:
            cnt += len(sentence.split())
    return cnt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_file', type=str, default="./data/polish/gpt-3.5-turbo/xsum_polish_gpt-3.5-turbo.raw_data.json")
    parser.add_argument('-s', type=int, default=10)
    args = parser.parse_args()
    print(count_words(args.input_file,args.s))