import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--train-json', type=str, required=True)
parser.add_argument('--train-csv', type=str, required=True)
args = parser.parse_args()

with open(args.coco_json) as json_reader:
    dataset = json.load(json_reader)

images = dataset['images']
annotations = dataset['annotations']
categories = dataset['categories']

