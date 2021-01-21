import numpy as np
from pycocotools import mask as mutils
from tqdm import tqdm

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


def save_to_csv(segm_json, output_csv, threshold):
    print(f'reading {segm_json}')
    with open(segm_json) as json_file:
        json_data = json.load(json_file)

    print(f'writing {output_csv}')
    with open(output_csv, 'w') as csv_writer:
        csv_writer.write('ImageId,EncodedPixels,Height,Width,CategoryId\n')
        for i in tqdm(range(len(json_data))):
            if json_data[i]['score'] < threshold:
                continue
            csv_writer.write('{},{},{},{},{}\n'.format(
                json_data[i]['image_id'],
                rle_to_string(rle_encode(mutils.decode(json_data[i]['segmentation']))),
                json_data[i]['segmentation']['size'][0],
                json_data[i]['segmentation']['size'][1],
                json_data[i]['category_id'],
            ))


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--segm-json', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.7)
    args = parser.parse_args()

    save_to_csv(args.segm_json, args.output_csv, args.threshold)

