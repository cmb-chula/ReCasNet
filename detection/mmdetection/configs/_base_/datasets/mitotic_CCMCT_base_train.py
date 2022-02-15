# dataset settings
dataset_type = 'MitoticDataset'
data_root = 'data/dataset/CCMCT/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadMitoticSlide', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PhotoMetricDistortion'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadMitoticSlide', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale= (512, 512),
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
    dict(type='LoadMitoticSlide', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale= (512, 512),
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='MitoticInferenceDataset',
            ann_file=[
                data_root + 'train.txt',
            ],
            img_prefix=[data_root],
            pipeline=train_pipeline)),
    val=dict(
        type='MitoticInferenceDataset',
        ann_file=data_root + 'val.txt',
        img_prefix=data_root ,
        pipeline=val_pipeline),
    test=dict(
        type='MitoticInferenceDataset',
        ann_file=data_root + 'inference_train.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='mAP')
