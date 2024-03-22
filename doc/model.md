# 模型介绍 [engine/model]
- 这个模型包含了 优化器、调度器、网络的创建与加载
- 这模型分为两个类 BaseModel 和 BaseGanModel，分别针对普通模型和gan模型  


# 模型使用 
- 针对 BaseModel介绍， BaseGanModel 相同
- BaseModel 成员变量
    - g_model 模型网络 比如 resnet
    - cfg [配置](config.md) 
    - device 当前的设备类型 cpu 或 gpu
- 继承BaseModel 实现以下方法
    - def \__init\__(self, cfg)
        - 初始化函数， 在这里面必须实现[损失函数](loss.md)的创建
        - params
            - cfg [配置](config.md)

    - def run_step(self, data, *, epoch=None, **kwargs)
        - 训练函数， 在这个里面要调用网络生成结果, 然后计算损失函数， 返回损失函数
        - params
            - data 数据， 用户自定义的数据处理，通过[数据增强](./data_aug.md) 的输出结果
            - epoch 当前的迭代次数
            - kwargs 用户自定义参数
        - 示例
            ```python
               def run_step(self, data, *, epoch=None, **kwargs):
                sample = data['input'].to(self.device)
                label = data['gt'].to(self.device)

                logits = self.g_model(sample)
                total_loss = self.pixel_loss(logits, label)
                return total_loss
            ```
    
    - def generator(self, data)
        - 推理时的方法， 返回推理结果
        - params
            - data 数据， 用户自定义的数据处理，通过[数据增强](./data_aug.md) 的输出结果
        - 示例
        ```python
            def generator(self, data):
                return self.g_model(sample)
        ```