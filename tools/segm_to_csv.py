import argparse
import json

from mmdet.datasets.kfashion import segm_to_csv

parser = argparse.ArgumentParser()
parser.add_argument('--segm-json', type=str, required=True)
parser.add_argument('--out-csv', type=str, required=True)
parser.add_argument('--bbox-thr', type=float, required=True)
parser.add_argument('--obj-min', type=int, default=None)
parser.add_argument('--obj-max', type=int, default=None)
args = parser.parse_args()

segm_json_results = json.load(open(args.segm_json))
segm_to_csv(segm_json_results, args.bbox_thr, args.out_csv, args.obj_min, args.obj_max)

