# 配置文件介绍[engine/config/defaults.py]

- 依赖 fvcore==0.1.5.post20210825
- 支持 yaml格式和python格式
- 配置文件分为 分为3个部分，分别是：
    - TRAINER
    - SOLVER
    - DATALOADER
    - OUTPUT_DIR 输出目录
    - OUTPUT_LOG_NAME 输出日子名

---
## TRAINER
- 训练器的配置参数

    |名称| 类型 | 描述|  
    |---|---|---|  
    |NAME| 字符串| 训练器的名称，默认是 BaseTrainer， 如果用户有特殊需求，必须继承BaseTrainer，按需求实现对应的方法 |  
    |ENABLE_EPOCH_METHOD| bool | 训练时数据的迭代方式 |
    | WEIGHTS | 字符串 | 预训练的权重路径，这个在训练时可以指定预训练的权重 |  
    | DEVICE | 字符串 | 训练时使用的设备 cpu or gpu ｜
    | PARADIGM | dict | 训练的模型，单机还是分布式 参见[paradigm](#paradigm)|
    | MODEL | dict | 训练时使用的模型, 参见[MODEL](#model)|  


#### MODEL
- 参数  
    |名称| 类型 | 描述|  
    |---|---|---|  
    |NAME | 字符串 | 使用的模型，用户必须继承BaseMode或者BaseGanModel实现对应虚函数|
    |GENERATOR|dict| 在 gan模型中时生成器网络的的参数，在普通模型中是网络的参数|
    |DISCRIMINATOR| dict | 只用在gan模型中才生效, 判别式网络的参数|     


#### PARADIGM [用户配置无效]  
- 参数  

    |名称| 类型 | 描述|  
    |---|---|---|  
    |TYPE| str | NORMAL: 普通模式 <br> DP: 数据并行 <br> DDP: 模型并行 |  
    |GPU_ID| int | 当前机器的GPU编号,相对于当前机器， TYPE=DDP 有效|  
    |GLOBAL_RANK| str | 当前机器的GPU编号,相对于所有机器， TYPE=DDP 有效 |  
    | world_size | int | GPU 总数|
    |NUM_PER_GPUS| int | 一台机器的GPU的总数 |    


---
## SOLVER
- 优化器和学习率调度器配置  
    |名称| 类型 | 描述|  
    |---|---|---|  
    |GRADIENT_ACCUMULATION_BATCH| bool | 是否启用梯度累加,默认[-1] 关闭 <br> 当模型或者数据量大时，且资源不够，可以启用梯度累加|  
    |TRAIN_PER_BATCH| int | 训练的 batch size |
    | TEST_PER_BATCH | int | 验证 batch size |
    | MAX_ITER | int | 最大 迭代次数 |
    | MAX_KEEP | int | checkpoint 的最大保存数量|
    |CHECKPOINT_PERIOD | int | 执行验证的周期 |
    | EMA | dict | [EMA](#ema) 的配置参数 |
    |GENERATOR | dict | 生成器的优化和[调度](#lr_scheduler)参数|
    |DISCRIMINATOR| dict | 判别器的优化和调度参数


#### EMA
- 参数  
    |名称| 类型 | 描述|  
    |---|---|---|  
    |ENABLED| bool | 开关，默认关闭 [false] |
    |DECAY_RATE| float | 移动平均的衰减率 |
    |num_updates| int | DECAY_RATE 调整周期, 默认[None] |

### LR_SCHEDULER
- 参数
    |名称| 类型 | 描述|  
    |---|---|---|  
    | ENABLED | bool | 开关 默认[false]关闭 |
    | TYPE | str | 调度器的名称 |
    |PARAMS | dict | 调度器的参数 |

- 支持的调度器
    - LRMultiplierScheduler
    - WarmupPolynomialDecay
    - pytorch中的所有的学习率调度器，比如[ExponentialLR, LinearLR, MultiStepLR]

### OPTIMIZER  
- 参数
    |名称| 类型 | 描述|  
    |---|---|---|  
    | TYPE | str | 优化器, 比如 Adam |
    |PARAMS | dict | 优化器的参数 |
    |CLIP_GRADIENTS| dict | 梯度裁剪的参数 |


--
## DATALOADER
- 数据相关的参数， 包括数据增强，数据加载与创建

# 运行参数
| 名称 | 类型| 描述 |
|---|---|---| 
| num-gpus |  int| 使用的GPU数量 |
| config-file| str | 配置文件路径 |  
| resume | |是否恢复训练|  
|distribute | | 是否尝试分布式训练 |
|num-machines| int | 机器总数 默认 1 |  
| machine-rank | int | 机架号 默认 0 |
| dist-url | str | 主机的 ip 和 端口 |
|opts |  | 可以在此处把 配置文件的值单独赋值 <br> --opts DATALOADER.NUM_WORKERS 2 |



