# 使用 engine  

##  BaseModel 使用说明：  

### 1. 继承BaseModel创建创基自己的模型
```python
from engine.model.base_model import BaseModel
from engine.model.build import BUILD_MODEL_REGISTRY
from engine.loss.pipe import LossKeyCompose
from engine.loss.pipe import LossCompose
import torch


@BUILD_MODEL_REGISTRY.register()
class MyModel(BaseModel):
    def __init__(self, cfg):
        super(MyModel, self).__init__(cfg)

        # 参见 engine/loss/README.md
        loss_cfg = cfg.TRAINER.LOSS
        if isinstance(loss_cfg, list):
            self.loss = LossCompose(loss_cfg)
        else:
            self.loss = LossKeyCompose(loss_cfg)
        return

    def create_model(self, params) -> torch.nn.Module:
        """
        根据自身情况考虑是否实现，推荐使用
        """
        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        此方法必须实现
        """
        input_data = data['input_data'].to(self.device, non_blocking=True)
        target_data = data['target_data'].to(self.device, non_blocking=True)

        fake_data = self.g_model(input_data)
        
        # 参见 engine/loss/README.md
        loss = self.loss((fake_data, target_data))

        return loss

    def generator(self, data):
        """
        此方法必须实现
        """
        scores = self.g_model(data.to(self.device, non_blocking=True))

        return scores
```    

### 2. 继承BaseTrainer定义自己的trainer, 实现你需要的方法
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

### 3. 定义自己的数据
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
 
    def __init__(self, my_param, transformer=None):
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
        results['color_fiedls'] = ['img']

        results = self.transformate(results)

        return {'input_data': F.to_tensor(results['img']), 'target_data': target_data}

    def __len__(self):
        return len(self.dataset)

```

### 4. 将上诉定义的包必须导入到程序中，必须！！！ 否则注册不生效

### 5. 定义训练入口, 开始训练
```python
from engine.schedule.scheduler import BaseScheduler
from codes import *

if __name__ == '__main__':
    BaseScheduler().schedule()
```   

## BaseGanModel 使用说明 针对GAN模型, 其他和BaseModel一致  

```python
from engine.model.gan_model import BaseGanModel
from engine.model.build import BUILD_MODEL_REGISTRY
from engine.loss.pipe import LossKeyCompose
from codes.backbone.discriminator import Discriminator
from codes.loss.gan_loss import GANLoss
import torch

@BUILD_MODEL_REGISTRY.register()
class RetouchGanModel(BaseGanModel):
    def __init__(self, cfg):
        super(RetouchGanModel, self).__init__(cfg)
        self.g_loss = LossKeyCompose(cfg.TRAINER.MODEL.LOSS, self.device)
        self.gan_loss = GANLoss(use_lsgan=True)
        return

    def create_d_model(self, params) -> torch.nn.Module:
        return Discriminator()

    def dmodel_step(self, data, epoch=None, **kwargs):
        img_tensor = data['img'].to(self.device)
        gt_tensor = data['gt'].to(self.device)

        fake_mask = self.g_model(img_tensor)

        output_real_loss = self.dmodel_forward(dict(d_model1=dict(input_tensor=gt_tensor)))
        output_fake_loss = self.dmodel_forward(dict(d_model1=dict(input_tensor=fake_mask.detach())))

        d_loss = self.gan_loss(output_real_loss['d_model1'], True) + self.gan_loss(output_fake_loss['d_model1'], False)

        total_loss = dict(d_model1=d_loss)
        return total_loss

    def gmodel_step(self, data, epoch=None, **kwargs):
        img_tensor = data['img'].to(self.device)
        gt_tensor = data['gt'].to(self.device)

        fake = self.g_model(img_tensor)

        loss_input = (fake, gt_tensor)

        total_loss = self.g_loss(dict(loss1=loss_input))

        return total_loss

    def generator(self, data):
        """
        :param data:
        :return:
        """
        return self.g_model(data.to(self.device))

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
        discriminator = [
            dict(name='dmodel_name1', params=dict(name='Discriminator')), #必须和solver匹配
            dict(name='dmodel_name2', params=dict(name='Discriminator'))
        ],
        # 详见 engine/loss/README.md
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
        ),
        g_step=1, #表示 g_model 训练间隔，只在 GAN中生效    
    ),
)
```

### 3. 优化器配置
```python
solver = dict(
    train_per_batch=8,
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
    ),
    discriminator = [
        dict(name='dmodel_name1', 
             params=dict(lr_scheduler=dict(
                                            enabled=False,
                                            type='MultiStepLR',
                                            params=dict(
                                                milestones=[80000, 100000, 125000],
                                                gamma=0.1
                                            )
                         ), 
                         optimizer=dict(
                                        type='AdamW',
                                        params=dict(
                                            weight_decay=0.0001,
                                            betas=[0.9, 0.999],
                                            lr=0.0001
                                        ),
                                        clip_gradients=dict(
                                            enabled=False,
                                        )                         
                        )
            )
        ),
        dict(name='dmodel_name2', 
             params=dict(lr_scheduler=dict(
                                            enabled=False,
                                            type='MultiStepLR',
                                            params=dict(
                                                milestones=[80000, 100000, 125000],
                                                gamma=0.1
                                            )
                         ), 
                         optimizer=dict(
                                        type='AdamW',
                                        params=dict(
                                            weight_decay=0.0001,
                                            betas=[0.9, 0.999],
                                            lr=0.0001
                                        ),
                                        clip_gradients=dict(
                                            enabled=False,
                                        )                         
                        )
            )
        ),    ]

)
output_dir = '/data'
output_log_name = 'my-log'

```