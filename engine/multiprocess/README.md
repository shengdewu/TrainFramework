# 使用 multiprocess

## 定义自己的多进程类
```python
from engine.multiprocess import MultiProcess


class MyMultiPorcess(MultiProcess):

    def execute(self, args=()):
        sources = args[0] # 进程处理的文件列表
        thread_number = args[1] # 进程号
        param = args[2]  # 外部传入的其他参数 必须是 list

        param1 = param[0]
        param2 = param[1]
        return
```    

## 调用方式
```python

    multi_process = MyMultiPorcess()
    multi_process.schedule(nthread=20, data_source_list=[], args=(param1, param2))

```

