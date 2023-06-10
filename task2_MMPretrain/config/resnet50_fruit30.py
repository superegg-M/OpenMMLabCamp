# 模型
model = dict(
    type='ImageClassifier',     # 主模型类型（对于图像分类任务，使用 `ImageClassifier`）
    backbone=dict(
        type='ResNet',          # 主干网络类型
        # 除了 `type` 之外的所有字段都来自 `ResNet` 类的 __init__ 方法
        # 可查阅 https://mmpretrain.readthedocs.io/zh_CN/latest/api/generated/mmpretrain.models.backbones.ResNet.html
        depth=50,
        num_stages=4,           # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(3, ),      # 输出的特征图输出索引。
        frozen_stages=-1,       # 冻结主干网的层数
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),    # 颈网络类型
    head=dict(
        type='LinearClsHead',         # 分类颈网络类型
        # 除了 `type` 之外的所有字段都来自 `LinearClsHead` 类的 __init__ 方法
        # 可查阅 https://mmpretrain.readthedocs.io/zh_CN/latest/api/generated/mmpretrain.models.heads.LinearClsHead.html
        num_classes=30,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0), # 损失函数配置信息
        topk=(1, 5),                 # 评估指标，Top-k 准确率， 这里为 top1 与 top5 准确率
    ),
    init_cfg=dict(type='Pretrained',checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth')
)

# 数据
dataset_type = 'CustomDataset'
# 预处理配置
data_preprocessor = dict(
    # 输入的图片数据通道以 'RGB' 顺序
    mean=[123.675, 116.28, 103.53],    # 输入图像归一化的 RGB 通道均值
    std=[58.395, 57.12, 57.375],       # 输入图像归一化的 RGB 通道标准差
    to_rgb=True,                       # 是否将通道翻转，从 BGR 转为 RGB 或者 RGB 转为 BGR
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='RandomResizedCrop', scale=224),     # 随机放缩裁剪
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # 随机水平翻转
    dict(type='PackInputs'),         # 准备图像以及标签
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='ResizeEdge', scale=256, edge='short'),  # 缩放短边尺寸至 256px
    dict(type='CenterCrop', crop_size=224),     # 中心裁剪
    dict(type='PackInputs'),                 # 准备图像以及标签
]

# 构造训练集 dataloader
train_dataloader = dict(
    batch_size=32,                     # 每张 GPU 的 batchsize
    num_workers=5,                     # 每个 GPU 的线程数
    dataset=dict(                      # 训练数据集
        type=dataset_type,
        data_root='data/training_set',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),   # 默认采样器
    persistent_workers=True,                             # 是否保持进程，可以缩短每个 epoch 的准备时间
)

# 构造验证集 dataloader
val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/val_set',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
# 验证集评估设置，使用准确率为指标， 这里使用 topk1 以及 top5 准确率
val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = val_dataloader  # test dataloader 配置，这里直接与 val_dataloader 相同
test_evaluator = val_evaluator    # 测试集的评估配置，这里直接与 val_evaluator 相同

# 训练策略
optim_wrapper = dict(
    # 使用 SGD 优化器来优化参数
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# 学习率参数的调整策略
# 'MultiStepLR' 表示使用多步策略来调度学习率（LR）。
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# 训练的配置，迭代 100 个 epoch，每一个训练 epoch 后都做验证集评估
# 'by_epoch=True' 默认使用 `EpochBaseLoop`,  'by_epoch=False' 默认使用 `IterBaseLoop`
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
# 使用默认的验证循环控制器
val_cfg = dict()
# 使用默认的测试循环控制器
test_cfg = dict()

# 通过默认策略自动缩放学习率，此策略适用于总批次大小 256
# 如果你使用不同的总批量大小，比如 512 并启用自动学习率缩放
# 我们将学习率扩大到 2 倍
auto_scale_lr = dict(base_batch_size=256)

# 运行设置
# 默认所有注册器使用的域
default_scope = 'mmpretrain'

# 配置默认的 hook
default_hooks = dict(
    # 记录每次迭代的时间。
    timer=dict(type='IterTimerHook'),

    # 每 100 次迭代打印一次日志。
    logger=dict(type='LoggerHook', interval=10),

    # 启用默认参数调度 hook。
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 每个 epoch 保存检查点。
    checkpoint=dict(type='CheckpointHook', interval=1),

    # 在分布式环境中设置采样器种子。
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 验证结果可视化，默认不启用，设置 True 时启用。
    visualization=dict(type='VisualizationHook', enable=False),
)

# 配置环境
env_cfg = dict(
   # 是否开启 cudnn benchmark
    cudnn_benchmark=False,

    # 设置多进程参数
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 设置分布式参数
    dist_cfg=dict(backend='nccl'),
)

# 设置可视化工具
vis_backends = [dict(type='LocalVisBackend')] # 使用磁盘(HDD)后端
visualizer = dict(
    type='UniversalVisualizer', vis_backends=vis_backends, name='visualizer')

# 设置日志级别
log_level = 'INFO'

# 从哪个检查点加载
load_from = None

# 是否从加载的检查点恢复训练
resume = False