_base_ = ["../_base_/default_runtime.py"]

# 基础设置
batch_size = 8
enable_amp = True

# 模型配置
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PT-v2m2",
        in_channels=4,
        num_classes=2,
        patch_embed_depth=1,
        patch_embed_channels=32,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(1, 1, 1, 1),
        enc_channels=(32, 64, 128, 256),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        drop_path_rate=0.1,
        grid_sizes=(0.15, 0.3, 0.6, 0.9),
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             weight=[1.0, 1.0],  # 根据实际类别比例调整
             loss_weight=1.0,
             ignore_index=-1),
    ],
)

# 训练计划
epoch = 30
eval_epoch = 30
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.005)

# 数据集配置
dataset_type = "aQcKITTIDataset"
data_root = "data/aqc"
ignore_index = -1
names = ["spreader", "cell_guide"]

# 根据点云范围调整的空间裁剪参数
point_cloud_range = (-25, -14.5, 3.5, 24.5, 11.5, 68)  # 对齐实际数据分布

data = dict(
    num_classes=2,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", p=0.5),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="GridSample",
                grid_size=0.05,
                mode="train",
                keys=("coord", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=point_cloud_range),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment")),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                mode="train",
                keys=("coord", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=point_cloud_range),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment")),
        ],
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="PointClip", point_cloud_range=point_cloud_range),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment")),
        ],
    ),
)