import numpy as np
import pandas as pd


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


def evaluation(gt_df, pred_df):
    eval_df = pd.merge(gt_df, pred_df, how='outer', on=['ImageId', 'CategoryId'], suffixes=['_gt', '_pred'])
    eval_df = eval_df[eval_df['ImageId'].isin(gt_df['ImageId'])]
    eval_df = eval_df[['ImageId', 'EncodedPixels_gt', 'Height_gt', 'Width_gt', 'CategoryId', 'EncodedPixels_pred']]
    eval_df = eval_df.rename(columns={'Height_gt': 'Height',
                                      'Width_gt': 'Width'})
    # IoU for True Positive
    idx_ = eval_df['EncodedPixels_gt'].notnull() & eval_df['EncodedPixels_pred'].notnull()
    IoU = eval_df[idx_].apply(calc_IoU_threshold, axis=1)

    # False Positive
    fp = (eval_df['EncodedPixels_gt'].isnull() & eval_df['EncodedPixels_pred'].notnull()).sum()

    # False Negative
    fn = (eval_df['EncodedPixels_gt'].notnull() & eval_df['EncodedPixels_pred'].isnull()).sum()
    print(eval_df[eval_df['EncodedPixels_gt'].notnull() & eval_df['EncodedPixels_pred'].isnull()])
    threshold_IoU = [0.5]
    scores = []
    for th in threshold_IoU:
        # True Positive
        tp = (IoU > th).sum()
        maybe_fp = (IoU <= th).sum()

        # False Positive (not Ground Truth) + False Positive (under IoU threshold)
        fp_IoU = fp + maybe_fp

        # Calculate evaluation score
        score = tp / (tp + fp_IoU + fn)
        scores.append(score)
        print(f"Threshold: {th}, Precision: {score}, TP: {tp}, FP: {fp_IoU}, FN: {fn}")

    mean_score = sum(scores) / len(threshold_IoU)
    print(f"Mean precision score: {mean_score}")
    return mean_score


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-csv', type=str, required=True)
    parser.add_argument('--pred-csv', type=str, required=True)
    args = parser.parse_args()

    print(f'reading gt_csv: {args.gt_csv}')
    gt = pd.read_csv(args.gt_csv)

    print(f'reading pred_csv: {args.pred_csv}')
    pred = pd.read_csv(args.pred_csv)

    print(f'evaluating...')
    evaluation(gt, pred)
