# 使用 engine  

## 1. 继承BaseModel创建创基自己的模型
```python
from engine.model.base_model import BaseModel
from engine.model.build import BUILD_MODEL_REGISTRY
import torch


@BUILD_MODEL_REGISTRY.register()
class MyModel(BaseModel):
    def __init__(self, cfg):
        super(MyModel, self).__init__(cfg)
        return

    def create_model(self, params) -> torch.nn.Module:
        '''
        :params 网络的参数 推荐dict类型
        
        eg: 
        已有网络：
        
        class MyNetwork(torch.nn.Module):
            def __init__(self, param1, param2):
                super(MyNetwork, self).__init__()
                return  
                
            def forward(self, x)
                return
                
                
        则 params = dict(
            name=MyNetwork,  # 自己的网络名
            param1=xxx,      #自己网络的参数
            param2=xxx, 
        )
        

        '''
        kwargs = dict()
        arch_name = ''
        for k, v in params.items():
            if k.lower() == 'name':
                arch_name = v
                continue
            kwargs[k.lower()] = v
            
        model = create_my_network(kwargs)
        
        return model

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        sample = data['input'].to(self.device)
        label = data['gt'].to(self.device)

        logits = self.g_model(sample)

        #计算损失
        total_loss = calculate(logits, label)

        self.g_optimizer.zero_grad()

        total_loss.backward()
        self.g_optimizer.step()

        return {'total_loss': total_loss.detach().item()}

    def generator(self, data):
        """
        :param data:
        :return:
        """

        logits = self.g_model(data['input'].to(self.device))
        return logits
```    

## 2. 继承BaseTrainer定义自己的trainer, 实现你需要的方法
```python
from engine.trainer.trainer import BaseTrainer
import logging
from engine.trainer.build import BUILD_TRAINER_REGISTRY
from engine.model.build import build_model
from engine.data.build import build_dataset

@BUILD_TRAINER_REGISTRY.register()
class MyTrainer(BaseTrainer):

    def __init__(self, cfg):
        super(MyTrainer, self).__init__(cfg)
        return

    def create_dataset(self, cfg):
        train_dataset = build_dataset(cfg.DATALOADER.TRAIN_DATA_SET)
        valid_dataset = build_dataset(cfg.DATALOADER.VAL_DATA_SET)
        return train_dataset, valid_dataset

    def create_model(self, cfg):
        #创建 MyModel
        return build_model(cfg)

    def before_loop(self):
        #训练前的一些处理
        return
    
    def after_loop(self):
        # 训练后的一些处理  
        self.model.disable_train()      
        return

    def iterate_after(self, epoch, loss_dict):
        #训练过程中的一些处理
        self.checkpoint.save(self.model, epoch)
        return

```  

## 4. 定义自己的数据
```python

from engine.transforms.pipe import TransformCompose
from engine.data.build import BUILD_DATASET_REGISTRY
import torch


@BUILD_DATASET_REGISTRY.register()
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, params, params2, transformers=None):
        super(MyDataSet, self).__init__()
        self.transformers = None
        
        # 数据增强
        if transformers is not None:
            self.transformers = TransformCompose(transformers)
            
        return

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pass

```

## 5. 将上诉定义的包 导入到程序中，必须！！！ 否则注册不生效

## 6. 定义训练入口, 开始训练
```python
from engine.schedule.scheduler import BaseScheduler
from codes import *

if __name__ == '__main__':
    BaseScheduler().schedule()
```


## 5 配置说明, 支持 python和yaml
```python

transformer = [
    dict(name='MyTransFormer',
         param='my transformer param')
]


dataloader = dict(
    num_workers=1,
    train_data_set=dict(
        name='MyDataset',
        params1='my data set param',
        transformer=transformer,
    ),
    val_data_set=dict(
        name='MyDataset',
        params1='my data set param',
        transformer=transformer,
    )
)

trainer = dict(
    name='MyTrainer',
    device='cuda',
    weights='',
    model=dict(
        name='MyModel',
        generator=dict(
            name='NyNetwork',
            param1='my network param',
        )
    ),
    loss=[ # 不一定时这个形式，可以自定义设计
        dict(name='MyLoss',
             params='my loss param'
             )
    ]
)

lr_scheduler = dict(
    enabled=True,
    type='LRMultiplierScheduler',
    params=dict(
        lr_scheduler_param=dict(
            name='WarmupCosineLR',
            gamma=0.1,
            steps=[40000, 80000, 160000, 19000],
        ),
        warmup_factor=0.01,
        warmup_iters=1000
    )
)

optimizer = dict(
    type='SGD',
    params=dict(
        momentum=0.9,
        lr=0.005,
        weight_decay=5E-4,
    ),
    clip_gradients=dict(
        enabled=False,
    ),
    g_step=1
)

solver = dict(
    train_per_batch=8,
    test_per_batch=2,
    max_iter=200000,
    max_keep=20,
    checkpoint_period=5000,
    generator=dict(
        lr_scheduler=lr_scheduler,
        optimizer=optimizer
    ),
)

output_dir = '/output'
output_log_name = 'log_name'

```