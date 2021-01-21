# dataset settings
dataset_type = 'KFashionDataset'
data_root = '/nas/dylan.minu/dataset/kfashion/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize',
         img_scale=[(700, 700), (800, 800), (900, 900)],
         multiscale_mode='value', keep_ratio=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(800, 800)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(800, 800)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train_all=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train',
        pipeline=train_pipeline),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'seed12648430/train.json',
        img_prefix=data_root + 'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'seed12648430/val.json',
        img_prefix=data_root + 'train',
        pipeline=val_pipeline
    ),
    val_mini=dict(
        type=dataset_type,
        ann_file=data_root + 'seed0_mini/val.json',
        img_prefix=data_root + 'train',
        pipeline=val_pipeline
    ),
    test_val=dict(
        type=dataset_type,
        ann_file=data_root + 'seed12648430/val.json',
        img_prefix=data_root + 'train',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_pubilc.json',
        img_prefix=data_root + 'test',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric=['proposal', 'bbox', 'segm'])
