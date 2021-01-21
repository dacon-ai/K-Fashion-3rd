import json
import os
from collections import OrderedDict
from collections import defaultdict
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd
from mmcv.utils import print_log
from pycocotools import mask as mutils

from .builder import DATASETS
from .coco import CocoDataset

""" DACON UTILS """


# https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# Used only for testing.
# This is copied from https://www.kaggle.com/paulorzp/run-length-encode-and-decode.
# Thanks to Paulo Pinto.
def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


def segm_to_csv(segm_json_results, bbox_threshold, output_csv, min_obj_per_image=None, max_obj_per_image=None):
    image_segms = defaultdict(list)
    for segm in segm_json_results:
        image_id = segm['image_id']
        image_segms[image_id].append(segm)

    with open(output_csv, 'w') as csv_writer:
        csv_writer.write('ImageId,EncodedPixels,Height,Width,CategoryId\n')
        for image_id, segms in image_segms.items():
            segms.sort(key=lambda x: x['score'], reverse=True)
            for i, segm in enumerate(segms):
                score = segm['score']
                if min_obj_per_image is None:
                    if score < bbox_threshold:
                        continue
                elif i >= min_obj_per_image and score < bbox_threshold:
                    continue

                if max_obj_per_image is None:
                    if score < bbox_threshold:
                        continue
                elif i >= max_obj_per_image:
                    continue

                csv_writer.write('{},{},{},{},{}\n'.format(
                    segm['image_id'],
                    rle_to_string(rle_encode(mutils.decode(segm['segmentation']))),
                    segm['segmentation']['size'][0],
                    segm['segmentation']['size'][1],
                    segm['category_id'],
                ))


        # for res in segm_json_results:
        #     if res['score'] < bbox_threshold:
        #         continue
        #     csv_writer.write('{},{},{},{},{}\n'.format(
        #         res['image_id'],
        #         rle_to_string(rle_encode(mutils.decode(res['segmentation']))),
        #         res['segmentation']['size'][0],
        #         res['segmentation']['size'][1],
        #         res['category_id'],
        #     ))


def segm_to_dataframe(segm_json_results, min_bbox_score=1.0):
    rows = list()
    for res in segm_json_results:
        if res['score'] < min_bbox_score:
            continue
        rows.append((
            res['image_id'],
            rle_to_string(rle_encode(mutils.decode(res['segmentation']))),
            res['segmentation']['size'][0],
            res['segmentation']['size'][1],
            res['category_id'],
            res['score'],
        ))
    return pd.DataFrame(rows, columns=['ImageId', 'EncodedPixels', 'Height', 'Width', 'CategoryId', 'Score'])


def calc_IoU(A, B):
    AorB = np.logical_or(A, B).astype('int')
    AandB = np.logical_and(A, B).astype('int')
    IoU = AandB.sum() / AorB.sum()
    return IoU


def rle_to_mask(rle_list, SHAPE):
    tmp_flat = np.zeros(SHAPE[0] * SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i, v in zip(strt, length):
            tmp_flat[(int(i) - 1):(int(i) - 1) + int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask


def calc_IoU_threshold(data):
    # Note: This rle_to_mask should be called before loop below for speed-up! We currently implement here to reduse memory usage.
    mask_gt = rle_to_mask(data['EncodedPixels_gt'].split(), (int(data['Height']), int(data['Width'])))
    mask_pred = rle_to_mask(data['EncodedPixels_pred'].split(), (int(data['Height']), int(data['Width'])))
    return calc_IoU(mask_gt, mask_pred)


def parallelize(data, func):
    num_cores = cpu_count()
    data_split = np.array_split(data, num_cores, axis=1)
    pool = Pool(num_cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def evaluation(df_gt, df_pred, bbox_threshold, iou_threshold=0.5):
    # https://www.kaggle.com/kyazuki/calculate-evaluation-score

    eval_df = pd.merge(df_gt, df_pred[df_pred['Score'] >= bbox_threshold],
                       how='outer', on=['ImageId', 'CategoryId'], suffixes=['_gt', '_pred'])
    eval_df = eval_df[eval_df['ImageId'].isin(df_gt['ImageId'])]
    eval_df = eval_df[
        ['ImageId', 'EncodedPixels_gt', 'Height_gt', 'Width_gt', 'CategoryId', 'EncodedPixels_pred']]
    eval_df = eval_df.rename(columns={'Height_gt': 'Height',
                                      'Width_gt': 'Width'})

    # IoU for True Positive
    idx_ = eval_df['EncodedPixels_gt'].notnull() & eval_df['EncodedPixels_pred'].notnull()
    IoU = eval_df[idx_].apply(calc_IoU_threshold, axis=1)
    # IoU = dd.from_pandas(eval_df[idx_], npartitions=16).\
    #     map_partitions(lambda df: df.apply((lambda row: calc_IoU_threshold(row)), axis=1)).\
    #     compute(scheduler='processes')

    # False Positive
    fp = (eval_df['EncodedPixels_gt'].isnull() & eval_df['EncodedPixels_pred'].notnull()).sum()

    # False Negative
    fn = (eval_df['EncodedPixels_gt'].notnull() & eval_df['EncodedPixels_pred'].isnull()).sum()

    # True Positive
    tp = (IoU > iou_threshold).sum()

    # False Positive (not Ground Truth) + False Positive (under IoU threshold)
    maybe_fp = (IoU <= iou_threshold).sum()
    fp_IoU = fp + maybe_fp

    # Calculate evaluation score
    score = tp / (tp + fp_IoU + fn)
    print_log(f"Dacon_mAP@[IoU={iou_threshold:.2f} | BBox={bbox_threshold:.2f}] = "
              f"{score:.6f} (TP={tp}, FP_GT={fp}, FP_IoU={maybe_fp}, FN={fn}")
    return score


@DATASETS.register_module()
class KFashionDataset(CocoDataset):
    CLASSES = ('top', 'blouse', 't-shirt', 'Knitted fabri', 'shirt', 'bra top', 'hood',
               'blue jeans', 'pants', 'skirt', 'leggings', 'jogger pants', 'coat', 'jacket',
               'jumper', 'padding jacket', 'best', 'kadigan', 'zip up', 'dress', 'jumpsuit')

    def ____parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x, y, w, h = ann['bbox']
            inter_w = max(0, min(x + w, img_info['width']) - max(x, 0))
            inter_h = max(0, min(y + h, img_info['height']) - max(y, 0))
            if inter_w * inter_h == 0:
                continue

            x1, y1, x2, y2 = x, y, x + w, x + y
            x1 = max(0, min(x1, img_info['width']))
            y1 = max(0, min(y1, img_info['height']))
            x2 = max(0, min(x2, img_info['width']))
            y2 = max(0, min(y2, img_info['height']))
            area = (x2 - x1) * (y2 - y1)
            if area <= 0 or (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x2, y2]
            segmentation = ann.get('segmentation', None)
            if segmentation:
                for segm in segmentation:
                    assert len(segm) % 2 == 0
                    for j in range(len(segm)):
                        if j % 2 == 0:  # x
                            segm[j] = max(0, min(segm[j], img_info['width']))
                        else:  # y
                            segm[j] = max(0, min(segm[j], img_info['height']))

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(segmentation)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        data_infos = super().load_annotations(ann_file)
        for x in data_infos:
            x['filename'] = '{}/{}'.format(x['filename'][0], x['filename'])
        return data_infos

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        if jsonfile_prefix:
            os.makedirs(os.path.dirname(jsonfile_prefix), exist_ok=True)
        result_files, tmp_dir = super().format_results(results, jsonfile_prefix, **kwargs)
        bbox_threshold = kwargs.get('bbox_threshold', 0.5)
        segm_json_results = json.load(open(result_files['segm']))
        segm_to_csv(segm_json_results, bbox_threshold,
                    result_files['segm'].replace('.segm.json', f'_thr{bbox_threshold}_segm.csv'))
        return result_files, tmp_dir

    def evaluate_dacon(self, results):
        bbox_thresholds = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)

        bbox_json_results, segm_json_results = self._segm2json(results)
        df_gt = pd.read_csv(self.ann_file.replace('.json', '.csv'))
        df_pred = segm_to_dataframe(segm_json_results, min(bbox_thresholds))

        eval_result = OrderedDict()
        for bbox_threshold in bbox_thresholds:
            score = evaluation(df_gt, df_pred, bbox_threshold)
            eval_result[f'Dacon_mAP@{bbox_threshold:.2f}'] = round(score, 6)
        return eval_result

    def evaluate(self, results, **kwargs):
        kwargs.update(
            classwise=True,
        )
        eval_results = super().evaluate(results, **kwargs)
        dacon_results = self.evaluate_dacon(results=results)
        eval_results.update(dacon_results)
        return eval_results
