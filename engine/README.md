# 使用 engine  

## 1. trainer
```python
import engine.trainer.trainer as trainer
class MyTrainer(trainer.BaseTrainer):
    def __init__(self, cfg):
        super(MyTrainer, self).__init__(cfg)
        setup_logger(cfg.OUTPUT_DIR, comm.get_rank(), name=__name__)

        train_dataset, test_dataset = create_dataset()

        return

    def create_model(self, cfg) -> BaseModel:
        return build_model(cfg)

    def loop(self):
        self.model.enable_train()

        for epoch in range(self.start_iter, self.max_iter):
            data = next(self.iter_train_loader)

            loss = self.model(data, epoch=epoch)

            addtion = self.model.get_addition_state_dict()
            self.checkpoint.save(epoch, self.model.get_state_dict(), **addtion)

        addtion = self.model.get_addition_state_dict()
        self.checkpoint.save(self.max_iter, self.model.get_state_dict(), **addtion)

```  

## 2. schedule  
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
