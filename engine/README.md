# 使用 engine  

## 1. 继承BaseModel创建创基自己的模型
```python
from engine.model.base_model import BaseModel
class MyModel(BaseModel):
    def __init__(self, cfg):
        super(MyModel, self).__init__(cfg)
        #创建损失函数以及其他你需要的
        self.your_loss = your_loss()
        return

    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        #你的训练
        #1. 取出输入数据和样本
        sample = data['input'].to(self.device)
        label = data['gt'].to(self.device)

        #3. 模型输出
        logits = self.g_model(sample)

        #4. 计算损失函数
        total_loss = self.your_loss(logits, label)

        #5. 优化
        self.g_optimizer.zero_grad()
        total_loss.backward()
        self.g_optimizer.step()

        return {'total_loss': total_loss.detach().item()}

    def generator(self, data):
        """
        :param data:
        :return:
        """
        # 你的推理
        logits = self.g_model(data['input'].to(self.device))

        return {'mask': mask, 'acc': acc}
        
    def create_g_model(self, cfg) -> torch.nn.Module:
        #创建你的模型
        model = create_your_model()
        return model

    def enable_distribute(self, cfg):
        # 一般不需要重写，除非使用了Gan模型
        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            logging.getLogger(__name__).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.g_model)
            self.g_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
        elif cfg.MODEL.TRAINER.TYPE == 0:
            logging.getLogger(__name__).info('launch model by parallel')
            self.g_model = torch.nn.parallel.DataParallel(self.g_model)
        else:
            logging.getLogger(__name__).info('launch model by singal machine')
        return   
```    

## 2. 继承BaseTrainer定义自己的trainer, 实现你需要的方法
```python
from engine.trainer.trainer import BaseTrainer

class MyTrainer(BaseTrainer):

    def __init__(self, cfg):
        super(MyTrainer, self).__init__(cfg)
        return

    def create_dataset(self, cfg):
        train_dataset = create_train_set() # 你自己的数据集
        valid_dataset = create_valid_set() #

        return train_dataset, valid_dataset

    def create_model(self, cfg):
        #创建你的模型
        return MyModel()

    def after_loop(self):
        #整个训练完成后你想要做的事
        return

    def iterate_after(self, epoch, loss_dict):
        #指定 CHECKPOINT_PERIOD 做的事情
        self.checkpoint.save(self.model, epoch)
        return

```  

## 3. 继承BaseScheduler， 实现 lunch_func方法
```python
from engine.schedule.scheduler import BaseScheduler
import MyTrainer

class MyScheduler(BaseScheduler):
    def __init__(self):
        super(AdaptiveScheduler, self).__init__()
        return

    def lunch_func(self, cfg, args):
        trainer = MyTrainer(cfg)
        trainer.resume_or_load(args.resume)
        trainer.loop()
        return
```

## 4 开始你的训练流程
```python
    MyScheduler().schedule()
```

## 5 配置说明
```python
DATALOADER: # 数据加载 包括 数据变化相关的配置放在此配置下
  NUM_WORKERS: 15 #数据加载使用的进程数
MODEL:
  TRAINER:
    MODEL: MyModel # 你的模型名
  DEVICE: cuda
  WEIGHTS: '' # 模型权重路径 用于预训练
OUTPUT_DIR:  '' #模型、日志输出路径
SOLVER:
  CHECKPOINT_PERIOD: 5000 # 模型保存周期
  LR_SCHEDULER: # 学习率调整方式
    ENABLED: true # 是否启动学习率调整
    GAMMA: 0.1
    LR_SCHEDULER_NAME: WarmupCosineLR
    STEPS:
    - 100000
    - 180000
    - 240000
    WARMUP_FACTOR: 0.01
    WARMUP_ITERS: 500
  MAX_ITER: 250000 # 训练迭代次数
  MAX_KEEP: 30 # 最大保存的模型数量
  OPTIMIZER: # 优化器配置
    GENERATOR:
      TYPE: 'AdamW' #优化器名 目前支持 pytorch自带的优化器
      PARAMS: # 优化器参数
        LR: 0.0001
  TEST_PER_BATCH:  # 测试的batch size
  TRAIN_PER_BATCH: 8 #训练的batch size
```