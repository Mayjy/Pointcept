_base_ = ["../_base_/default_runtime.py"]

# 基础设置
batch_size = 12
num_worker = 12
enable_amp = True

ignore_index = 0 # 根据任务需求调整

# 模型设置
model = dict(
    type="DefaultSegmentorV2",
    num_classes=2,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_flash=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0),
    ],
)

# 训练设置
epoch = 300
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.006,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# 数据集设置
dataset_type = "aQcKITTIDataset"
data_root = "data/aqc"

data = dict(
    num_classes=2,
    ignore_index=ignore_index,
    names=["spreader", "cell_guide"],
    train=dict(
        type="aQcKITTIDataset",
        split=["train"],
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment"),
                return_grid_coord=True,
            ),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(  # 添加验证集配置
        type="aQcKITTIDataset",
        split=["val"],  # 验证集的分割名
        data_root=data_root,
        transform=[
            dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="test", keys=("coord", "segment")),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment")),
        ],
        test_mode=False,  # 验证集通常采用测试模式
        # test_cfg=dict(
        #     voxelize=dict(
        #         type="GridSample",  # ✅ 已注册的预处理类
        #         grid_size=0.05,
        #         hash_type="fnv",
        #         mode="test",
        #         return_grid_coord=True,  # ✅ 生成 grid_coord
        #         keys=("coord",),
        #     ),
        #     crop =None,
        #     post_transform=[
        #         dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
        #         dict(type="ToTensor"),
        #         dict(type="Collect", keys=("coord", "grid_coord", "index")),  # ✅ 引用 grid_coord
        #     ],
        #     collate_fn=dict(type="default_collate"),  # ⭐ 处理列表数据
        # ),
        ignore_index=ignore_index,
    ),
    # test=dict(
    #     type="AQCDataset",
    #     split="test",
    #     data_root=data_root,
    #     transform=[],  # ⚠️ 测试时不应用随机增强
    #     test_mode=True,
    #     test_cfg=dict(
    #         voxelize=dict(
    #             type="GridSample",  # ✅ 已注册的预处理类
    #             grid_size=0.05,
    #             hash_type="fnv",
    #             mode="test",
    #             return_grid_coord=True,  # ✅ 生成 grid_coord
    #             keys=("coord",),
    #         ),
    #         post_transform=[
    #             dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
    #             dict(type="ToTensor"),
    #             dict(type="Collect", keys=("coord", "grid_coord", "index")),  # ✅ 引用 grid_coord
    #         ],
    #         collate_fn=dict(type="default_collate"),  # ⭐ 处理列表数据
    #     ),
    #     ignore_index=ignore_index,
    # ),
)
