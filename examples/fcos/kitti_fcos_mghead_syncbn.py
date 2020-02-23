import itertools
import logging


# norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["Car",],),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings

train_cfg = dict()

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
    ),
    score_threshold=0.05,
    post_center_limit_range=[0, -40.0, -5.0, 70.4, 40.0, 5.0],
    max_per_img=100,
)

# dataset settings
dataset_type = "KittiDataset"
data_root = "/media/zhengjs/225A6D42D4FA828F/kitti_object/"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path=data_root + "/dbinfos_train.pkl",
    sample_groups=[dict(Car=15,),],
    db_prep_steps=[
        dict(filter_by_min_num_points=dict(Car=5,)),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[0.25, 0.25, 0.25],
    gt_rot_noise=[-0.15707963267, 0.15707963267],
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.0, 0.0, 0.0],
    remove_points_after_sample=True,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[0, -38.4, -3, 67.2, 38.4, 1],
    voxel_size=[0.15, 0.15, 4.0],
    max_points_in_voxel=50,
    max_voxel_num=12000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile"),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    # dict(type="SegmentAssignTarget", cfg=None),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile"),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    # dict(type="SegmentAssignTarget", cfg=None),
    dict(type="Reformat"),
]

train_anno = data_root + "/kitti_infos_train.pkl"
val_anno = data_root + "/kitti_infos_val.pkl"
test_anno = None

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + "/kitti_infos_train.pkl",
        ann_file=train_anno,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root + "/kitti_infos_val.pkl",
        ann_file=val_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# model settings
model = dict(
    type="FCOS",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64],
        with_distance=False,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(
        type="PointPillarsScatter",
        ds_factor=1,
        norm_cfg=norm_cfg,
        conv_cfg=dict(
            type='ResNet',
            in_channels=64,
            depth=18,
            # groups=64,
            # base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            style='pytorch'
        ),
    ),
    neck=dict(
        type='FPN',
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=0,
        end_level=3,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=3,
        relu_before_extra_convs=True
    ),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=(4, 8, 16),
        regress_ranges=((-1, 3.0), (3.0, 6.0), (6.0, 25.0)),
        tasks=tasks,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        voxel_cfg=voxel_generator,
    ),
)

# optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# # learning policy
# lr_config = dict(
#     type='multinomial',
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150])

# optimizer
optimizer = dict(
    type="adam",
    amsgrad=0.0,
    wd=0.01,
    fixed_wd=True,
    moving_average=False,
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle",
    lr_max=3e-3,
    moms=[0.95, 0.85],
    div_factor=10.0,
    pct_start=0.4,
)

checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 50
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "/media/zhengjs/225A6D42D4FA828F/kitti_object/det3d"
load_from = None
resume_from = None
workflow = [("train", 5), ("val", 1)]
