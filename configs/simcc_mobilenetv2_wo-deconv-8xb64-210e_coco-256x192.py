_base_ = ['./_base_/default_runtime.py']

load_from = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192-4b0703bb_20221010.pth"

# runtime
train_cfg = dict(max_epochs=210, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

num_keypoints = 17

# runtime
max_epochs = 700
base_lr = 4e-3
train_batch_size = 8
val_batch_size = 8

backend_args = dict(backend='local')

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=train_cfg['max_epochs'],
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel', input_size=(192, 256), sigma=6.0, simcc_split_ratio=2.0)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(7, ),
        ),
    head=dict(
        type='SimCCHead',
        in_channels=1280,
        out_channels=17,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        deconv_out_channels=None,
        loss=dict(type='KLDiscretLoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),

    dict(
        type='GenerateTarget',
        encoder=codec,
        use_dataset_keypoint_weights=False),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]


# mapping
halpe_coco = [(i, i) for i in range(17)]



# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'dataset/'

# train datasets
dataset_halpe = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='annotations.json',
    data_prefix=dict(img='img/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe_coco)
    ],
)


# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=5,
    pin_memory=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            dataset_halpe,
        ],
        pipeline=train_pipeline,
        test_mode=False,
    ))

# val datasets
val_halpe = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='annotations.json',
    data_prefix=dict(img='img/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=num_keypoints,
            mapping=halpe_coco)
    ],
)


val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=5,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            val_halpe,
        ],
        pipeline=val_pipeline,
        test_mode=True,
    ))

test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='AUC', rule='greater', max_keep_ckpts=1))

# evaluators
test_evaluator = [dict(type='PCKAccuracy', thr=0.1), dict(type='AUC')]
val_evaluator = test_evaluator