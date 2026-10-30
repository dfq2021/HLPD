import json
import argparse
import os

def truncation(args):
    if not os.path.exists(args.output_path): os.makedirs(args.output_path)
    if args.mode == 1:
        for len in [30, 60, 90, 120, 150, 180]:
            with open(args.input_file,'r',encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                data[key] = [" ".join(i.split(" ")[:len]) for i in value]
            with open(os.path.join(args.output_path, args.input_file.split("/")[-1].split(".raw_data")[0] + f"_truncation_{len}.raw_data.json"),'w') as f:
                json.dump(data,f)
    else:
        with open(args.input_file,'r',encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            data[key] = value[:args.nums]
        with open(os.path.join(args.output_path, args.input_file.split("/")[-1].split(".raw_data")[0] + f"_{args.nums}.raw_data.json"),'w') as f:
            json.dump(data,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_file', type=str, default="./data/polish/gpt-3.5-turbo/xsum_polish_gpt-3.5-turbo.raw_data.json")
    parser.add_argument('-o', '--output_path', type=str, default="./data/truncation/")
    parser.add_argument('--mode', type=int, choices=[1, 2],default=1)
    parser.add_argument('--nums', type=int, default=10)
    args = parser.parse_args()
    truncation(args)