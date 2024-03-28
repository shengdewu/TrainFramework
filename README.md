<img src="doc/title.jpg" width="100">

TrainFramework 是一个简单的以pytorch为基础的训练框架， 里面包含了数据增强，数据加载，checkpoint，通用的损失函数模块<br>  

*所有模块都依赖[`fvcore.common.registry`]的注册机制*<br>  

*依赖 ubuntu18.04 或 ubuntu20.04*<br>  

## 目标  

实现了数据分布式和模型分布式训练、断点恢复训练  

简化模型开发过程，让用户专注于模型的设计，减少对优化器、迭代器、数据加载以及训练方式的关心  
<br>   

## 结构  

- `调度器` 负责解析配置，并确定训练方式[DP or DDP], 然后调度训练器来训练模型<br>

- `训练器` 负责管理数据、模型、checkpoint, 训练模型，与加载权重<br>

- `模型` 负责管理网络、优化器、学习率调度[他们的共同点是都有状态]<br>

    这个模型不是一般意义的模型，考虑到处理模型的状态，包括网络的状态，优化器的状态，学习率调度的状态， 把他们集合在一起方便管理, 所以这个模型就是三者的组合  


```none
    调度器
    ｜---训练器
        ｜
        ｜----数据
        ｜
        ｜----模型
        ｜    ｜---网络
        ｜    ｜---优化器
        ｜    ｜---学习率调度器
        ｜
        ｜
        ｜
        ｜----checkpoint

```

## 环境  

- [python3.6 或者 python3.7](https://www.python.org/downloads/source/)  

- [docker](https://docs.docker.com/engine/install/ubuntu/)安装  

- cuda102  [dockerfile](docker/Dockerfile.cu102) 支持 cuda 10  

- cuda111 [dockerfile](docker/Dockerfile) 支持 cuda 11  

- [依赖](docker/requirements.txt)  

 <br>  

## 编译与安装  
<br>  

## `Docker`  

###  编译训练引擎  

```python
git clone https://codeup.aliyun.com/601b69af841cc46b7c49ab5f/ai-lab/TrainFramework.git

python3 setup.py bdist_wheel

```

### 编译基础docker环境 [Dockerfile](docker/Dockerfile)
```python
docker build ./ -f docker/Dockerfile -t dl.nvidia/cuda:11.1-cudnn8-devel-torch.1.10
```

### 编译训练docker环境  

- 根据[简单的使用](#简单的使用)实现训练模块  

- 定义自己的dockerfile  

    ```python
    FROM dl.nvidia/cuda:11.1-cudnn8-devel-torch.1.10 # 第2步中生成的镜像
    COPY codes /home/train                           # 自己的训练代码
    COPY dist/engine_frame-*.whl  /home/whl          # 第1步生成的训练引擎
    RUN pip3 install /home/whl/engine_frame-*.whl
    WORKDIR /home/train
    ENTRYPOINT ["python3", "train.py"]    
    ```
- 编译自己的训练代码
    ```python
    docker build ./ -f Dockerfile -t train
    ```
- 训练 指定[运行参数](doc/config.md#运行参数)开始训练  
    ```none
    docker run --gpus='"device=0"' --shm-size=20g -v /mnt:/mnt -t train --config-file /mnt/config/train.py --num-gpus 1
    ```

## `Ubuntu`  

###  编译训练引擎  

```python
git clone https://codeup.aliyun.com/601b69af841cc46b7c49ab5f/ai-lab/TrainFramework.git

python3 setup.py bdist_wheel

```  

### 安装[依赖](docker/requirements.txt) 

```none
pip3 install -r docker/requirements.txt
```    

### 安装训练引擎  

```none
pip3 install engine_frame-xx.whl
```

<br>  

## 模块介绍
* [配置](doc/config.md)的使用
* [数据增强](doc/data_aug.md)的使用
* [模型](doc/model.md)的创建
* [损失函数](doc/loss.md)的使用  


## 简单的使用

#### 1. 继承BaseModel创建创基自己的模型
- 必须在构造函数里创建[损失函数](doc/loss.md), 推荐使用LossKeyCompose  
```python
from engine.model.base_model import BaseModel
from engine.model.build import BUILD_MODEL_REGISTRY
from engine.loss.pipe import LossKeyCompose

@BUILD_MODEL_REGISTRY.register()
class MyModel(BaseModel):
    def __init__(self, cfg):
        super(MyModel, self).__init__(cfg)
        self.loss = LossKeyCompose(cfg.TRAINER.LOSS)
        return

    def create_model(self, params) -> torch.nn.Module:
        """
        根据自身情况考虑是否实现，推荐不实现，使用默认的
        """
        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        此方法必须实现
        """
        input_data = data['input_data'].to(self.device, non_blocking=True)
        target_data = data['target_data'].to(self.device, non_blocking=True)

        fake_data = self.g_model(input_data)
        
        loss = self.loss((fake_data, target_data))

        return loss

    def generator(self, data):
        """
        此方法必须实现
        """
        scores = self.g_model(data.to(self.device, non_blocking=True))

        return scores
```    

#### 2. 继承BaseTrainer定义自己的trainer, 实现你需要的方法[`可以不实现`]
```python
from engine.trainer.trainer import BaseTrainer
from engine.trainer.build import BUILD_TRAINER_REGISTRY


@BUILD_TRAINER_REGISTRY.register()
class MyTrainer(BaseTrainer):
    """
    可以不实现，如果不关心中途结果
    """
    def __init__(self, cfg):
        super(MyTrainer, self).__init__(cfg)
        return

    def after_loop(self):
        """
        训练的后处理
        """
        return

    def iterate_after(self, epoch, loss_dict):
        """
        训练的中的处理
        """
        return
```  

#### 3. 定义自己的数据
```python
from engine.data.dataset import EngineDataSet
from engine.data.build import BUILD_DATASET_REGISTRY
import engine.transforms.functional as F
import cv2


__all__ = [
    'MyDataset'
]


@BUILD_DATASET_REGISTRY.register()
class MyDataset(EngineDataSet):
 
    def __init__(self, my_param, transformer:List):
        super(MyDataset, self).__init__(transformer)
        return

    def __getitem__(self, index):
        img_path, target_data = self.dataset[index]
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        results = dict()
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = results['img_shape']
        results['img_fields'] = ['img']
        results['color_fields'] = ['img']

        results = self.data_pipeline(results)

        return {'input_data': F.to_tensor(results['img']), 'target_data': target_data}

    def __len__(self):
        return len(self.dataset)

```

#### 4. 将上诉定义的包必须导入到程序中，必须！！！ 否则注册不生效

#### 5. 定义训练入口, 开始训练
```python
from engine.schedule.scheduler import BaseScheduler
from codes import *

if __name__ == '__main__':
    BaseScheduler().schedule()
```   


## 配置说明, 支持 python和yaml

### 1. 数据配置
```python

dataloader = dict(
    num_workers=8,
    train_data_set=dict(
        name='MyDataset',  # 训练数据名，以下使其初始化参数
        my_data_param='my_data_param',
        transformer=[
            dict(
                name='RandomFlip',
                direction=['horizontal', 'vertical', 'diagonal'],
                p=0.5
            ),
            dict(
                name='Resize',
                target_size=256,
                interpolation='INTER_LINEAR',
                keep_ratio=False,
                clip_border=True,
            ),
            dict(
                name='RandomColorJitter',
                brightness_limit=[0.6, 1.2],
                brightness_p=0.6,
                contrast_limit=[0.6, 1.4],
                contrast_p=0.6,
                saturation_limit=[0.6, 1.4],
                saturation_p=0.5,
                hue_limit=[-0.1, 0.1],
                hue_p=0.05,
                blur_limit=[3, 7],
                blur_p=0.2,
                gamma_limit=[0.3, 3.0],
                gamma_p=0.1,
            )
        ]
    ),
    val_data_set=dict(
        name='MyDataset',
        my_data_param='my_data_param',
        transformer=[
            dict(
                name='Resize',
                target_size=256,
                interpolation='INTER_LINEAR',
                keep_ratio=False,
                clip_border=True,
            ),
        ]
    )
)
```

### 2. 训练器配置
```python
trainer = dict(
    name='MyTrainer', #用用户自定义训练器名
    device='cuda',
    weights='', # 预训练模型路径
    model=dict(
        name='MyModel', #用户自定义模型名
        generator=dict(
            name='MyNetwork', #用户自定义的网络名
            my_network_param='my_network_param'
        ),
        loss=dict(
            loss1=[
                dict(name='CrossEntropyLoss', param=dict(lambda_weight=10.))
            ],
            loss2=[
                dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor', 'target_tensor'])
            ]
        ),
        ema=dict(
            enable=True,  # gender enable ema 只对 生成器有效
            decay_rate=0.995        
        )  
    ),
)
```

### 3. 优化器配置
```python
solver = dict(
    gradient_accumulation_batch=-1,  # 是否启用梯度累加 > 1 启用
    train_per_batch=8,  # 当 gradient_accumulation_batch > 1 时，真实的 train_per_batch = train_per_batch // gradient_accumulation_batch 
    test_per_batch=8,
    max_iter=200000,
    max_keep=20,
    checkpoint_period=5000,
    generator=dict(
        lr_scheduler=dict(
            enabled=True,
            type='LRMultiplierScheduler',  # 学习率调度方法
            params=dict(           #学习率调度方法对应的参数
                lr_scheduler_param=dict(
                    name='WarmupMultiStepLR',
                    gamma=0.1,
                    steps=[40000, 120000, 180000, 240000],
                ),
                max_iter=200000,
                warmup_factor=0.01,
                warmup_iter=500,
                warmup_method='linear'
            )
        ),
        optimizer=dict(
            type='Adam',  #优化器名
            params=dict(  #优化器参数
                lr=0.01,
                weight_decay=1e-6,
            ),
            clip_gradients=dict(
                enabled=False,
            )
        )
    )
)
output_dir = '/data'
output_log_name = 'my-log'

```


