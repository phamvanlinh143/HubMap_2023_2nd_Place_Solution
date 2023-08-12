
checkpoint_config = dict(interval=1, save_last=True, max_keep_ckpts=5, save_optimizer=False)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained_weights/cascade_mask_rcnn_coat_small_mstrain_480-800_giou_4conv1f_adamw_3x_coco_4f7a069e.pth'
resume_from = None
workflow = [('train', 1)]
#################################################################################

# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=0.0003, pct_start=0.1)

runner = dict(type='EpochBasedRunner', max_epochs=40)

fp16 = dict(loss_scale=dict(init_scale=512))

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05)

optimizer_config = dict(grad_clip=None)
##################################################################
NUM_CLASSES = 1
##################################################################
# model settings
# model settings
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='CoaT',        
        patch_size=4, 
        embed_dims=[152, 320, 320, 320], 
        serial_depths=[2, 2, 2, 2], 
        parallel_depth=6, 
        num_heads=8, 
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.2,
        out_features=["x1_nocls", "x2_nocls", "x3_nocls", "x4_nocls"],
        return_interm_layers=True,
        ),
    neck=dict(
        type='FPN',
        in_channels=[152, 320, 320, 320],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=NUM_CLASSES,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg = dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=250,
            mask_thr_binary=0.5)))

#########################################################################

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type='RandomRotate90', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ElasticTransform', alpha=120, sigma=6.0,alpha_affine=3.5999999999999996, p=1.0),
            dict(type='GridDistortion', p=1.0),
            dict(type='OpticalDistortion', distort_limit=2, shift_limit=0.5, p=1.0),
        ],
        p=0.5),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0,
        scale_limit=(-0.2, 0.2),
        rotate_limit=(-5, 5),
        interpolation=1,
        border_mode=0,
        value=(0, 0, 0),
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.35,
        contrast_limit=0.5,
        brightness_by_max=True,
        p=0.5),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=30,
        sat_shift_limit=30,
        val_shift_limit=0,
        p=0.5),
    dict(type='ImageCompression', quality_lower=90, quality_upper=95, p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussNoise', var_limit=(0, 50.0), mean=0, p=1.0),
            dict(type='GaussianBlur', blur_limit=(3, 7), p=1.0),
            dict(type='Blur', blur_limit=3, p=1.0)
        ],
        p=0.5)
]


###########################################################################################################
# augmentation strategy originates from DETR / Sparse RCNN
stn_aug_root = "datasets/stain_9tiles_augs/"
stn_img_ext = ".tif"
margin = 128
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='StainTransform', 
         img_aug_root=stn_aug_root, 
         img_ext=stn_img_ext,
         margin=margin,
         prob=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
        dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[
                           (800, 1333), (832, 1333), (864, 1333), (896, 1333), (928, 1333),
                            (960, 1333), (992, 1333), (1024, 1333), (1056, 1333),
                           (1088, 1333), (1120, 1333)
                           ],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(900, 1333), (1000, 1333), (1100, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(768, 900),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[
                           (800, 1333), (832, 1333), (864, 1333), (896, 1333), (928, 1333),
                            (960, 1333), (992, 1333), (1024, 1333), (1056, 1333),
                           (1088, 1333), (1120, 1333)
                                     ],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]


train_same_wsi_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='StainTransform', 
         img_aug_root=stn_aug_root, 
         img_ext=stn_img_ext,
         margin=margin,
         prob=1.0),
    dict(type='RandomFlip', flip_ratio=0.5),
        dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[
                           (800, 1333), (832, 1333), (864, 1333), (896, 1333), (928, 1333),
                            (960, 1333), (992, 1333), (1024, 1333), (1056, 1333),
                           (1088, 1333), (1120, 1333)
                           ],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(900, 1333), (1000, 1333), (1100, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(768, 900),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[
                           (800, 1333), (832, 1333), (864, 1333), (896, 1333), (928, 1333),
                            (960, 1333), (992, 1333), (1024, 1333), (1056, 1333),
                           (1088, 1333), (1120, 1333)
                                     ],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 1024),
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
###########################################################################################################

classes = ("blood_vessel",)
dataset_type = 'CocoDataset'
data_root = 'datasets/'

train_set1 = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds1/ds1_wsi1_left_train.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=train_pipeline)

train_set2 = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds1/ds1_wsi1_right_train.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=train_pipeline)

train_set3 = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds1/ds1_wsi2_right_train.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=train_same_wsi_pipeline)

train_set4 = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds1/ds1_wsi1_ignore.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=train_pipeline)

train_set5 = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds2/ds2_wsi1.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=train_pipeline)

train_set6 = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds2/ds2_wsi2.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=train_same_wsi_pipeline)

train_set7 = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds2/ds2_wsi3.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=train_pipeline)

train_set8 = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds2/ds2_wsi4.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=train_pipeline)


val_set = dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds1/ds1_wsi2_left_val.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=test_pipeline)

test_set = test=dict(
        type=dataset_type,
        ann_file=data_root + 'hm_9tiles_crop128_1cls/ds1/ds1_wsi2_left_val.json',
        img_prefix=data_root + 'train_9tiles_crop128/',
        classes=classes,
        pipeline=test_pipeline)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=[train_set1, train_set2, train_set3,
           train_set4, train_set5, train_set6,
           train_set7, train_set8],
    val=val_set,
    test=test_set,
    persistent_workers=True)

evaluation = dict(save_best='auto', rule="greater", interval=1, metric=['segm', 'bbox'])

work_dir = 'workdirs/9tiles_crop128/coat_small/ds1_w2l_pt/'

