_base_ = ["../_base_/default_runtime.py"]

# 基础设置
batch_size = 9# 每个批次的样本数量
num_worker = 9# 数据加载时使用的工作线程数量
#enable_amp = True # 是否启用自动混合精度
enable_amp = False

ignore_index = -1 # 忽略的标签索引，通常用于忽略背景或无效标签

# 模型设置
model = dict(
    type="DefaultSegmentorV2",  # 模型类型
    num_classes=2,  # 分类数量
    backbone_out_channels=64,  # 主干网络输出通道数
    backbone=dict(
        type="PT-v3m1",  # 主干网络类型
        in_channels=4,  # 输入通道数
        order=("z", "z-trans", "hilbert", "hilbert-trans"),  # 编码顺序
        stride=(2, 2, 2, 2),  # 步幅
        enc_depths=(2, 2, 2, 4, 1),  # 编码器深度
        enc_channels=(32, 64, 128, 256, 512),  # 编码器通道数
        enc_num_head=(2, 4, 8, 16, 32),  # 编码器头数
        enc_patch_size=(256, 256, 256, 256, 256),  # 编码器补丁大小
        dec_depths=(2, 2, 2, 2),  # 解码器深度
        dec_channels=(64, 64, 128, 256),  # 解码器通道数
        dec_num_head=(4, 4, 8, 16),  # 解码器头数
        dec_patch_size=(256, 256, 256, 256),  # 解码器补丁大小
        mlp_ratio=4,  # MLP 比例
        qkv_bias=True,  # 是否使用 QKV 偏置
        drop_path=0.3,  # 丢弃路径概率
        shuffle_orders=False,  # 是否打乱顺序
        pre_norm=True,  # 是否使用预归一化
        enable_flash=True,  # 是否启用闪存
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0,ignore_index = -1),  # 交叉熵损失及其权重
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0,ignore_index = -1),  # Lovasz 损失及其权重
    ],
)

# 训练设置
epoch = 300  # 训练的总轮数
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)  # 优化器类型及其参数
scheduler = dict(
    type="OneCycleLR",  # 学习率调度器类型
    max_lr=0.006,  # 最大学习率
    pct_start=0.05,  # 学习率上升阶段的比例
    anneal_strategy="cos",  # 学习率退火策略
    div_factor=10.0,  # 初始学习率除数
    final_div_factor=1000.0,  # 最终学习率除数
)

# 数据集设置
dataset_type = "aQcKITTIDataset"
data_root = "data/aqc"

data = dict(
    num_classes=2,  # 分类数量
    ignore_index=ignore_index,  # 忽略的标签索引
    names=["spreader", "cell_guide"],  # 类别名称
    train=dict(
        type="aQcKITTIDataset",  # 训练集数据集类型
        split=["train"],  # 训练集分割
        data_root=data_root,  # 数据集根目录
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),  # 随机旋转
            dict(type="RandomScale", scale=[0.9, 1.1]),  # 随机缩放
            dict(type="RandomFlip", p=0.5),  # 随机翻转
            dict(type="RandomJitter", sigma=0.005, clip=0.02),  # 随机抖动
            dict(
                type="GridSample",
                grid_size=0.2,
                hash_type="fnv",
                mode="train",
                keys=("coord", "segment", "strength"),
                return_grid_coord=True,
            ),  # 网格采样
            dict(type="PointClip", point_cloud_range=(-25, -14.5, 3.5, 24.5, 11.5, 68)),
            dict(type="ToTensor"),  # 转换为张量
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
                # feat_keys=("coord"),
            ),  # 数据收集
        ],
        test_mode=False,  # 是否为测试模式
    ),
    val=dict(
        type="aQcKITTIDataset",  # 验证集数据集类型
        split=["val"],  # 验证集分割
        data_root=data_root,  # 数据集根目录
        transform=[
            dict(
                type="GridSample", 
                grid_size=0.2, 
                hash_type="fnv", 
                mode="train", 
                keys=("coord", "segment", "strength"),
                return_grid_coord=True
            ),  # 网格采样
            dict(type="PointClip", point_cloud_range=(-25, -14.5, 3.5, 24.5, 11.5, 68)),
            dict(type="ToTensor"),  # 转换为张量
            dict(
                type="Collect", 
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
                # feat_keys=("coord"),
            ),  # 数据收集
        ],
        test_mode=False,  # 是否为测试模式
    ),
    test=dict(
        type="aQcKITTIDataset",
        split="test",
        data_root=data_root,
        transform=[],  # ⚠️ 测试时不应用随机增强
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",  # ✅ 已注册的预处理类
                grid_size=0.2,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,  # ✅ 生成 grid_coord
                keys=("coord", "strength"),
            ),
            crop=None,
            post_transform=[
                dict(type="PointClip", point_cloud_range=(-25, -14.5, 3.5, 24.5, 11.5, 68)),
                dict(type="ToTensor"),
                dict(
                    type="Collect", 
                    keys=("coord", "grid_coord","index"),
                    feat_keys=("coord", "strength"),
                    # feat_keys=("coord"),
                ),  # ✅ 引用 grid_coord
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1, 1]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)
