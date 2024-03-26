#  损失函数说明[engine/loss]  

支持自定义和一部分pytorch的损失函数  
 - [x] L1Loss  
 - [x]  NLLLoss  
 - [x]  MSELoss  
 - [x]  BCELoss  
 - [x]  BCEWithLogitsLoss  
 - [x]  SmoothL1Loss  
 - [x]  SoftMarginLoss  
 - [x]  CrossEntropyLoss  
 - [x]  MultiLabelSoftMarginLoss  
 - [x]  TripletMarginLoss  
 - [x]  TripletMarginWithDistanceLoss  
 - [x]  BinaryDiceLoss
 - [x]  GeneralizedDiceLoss
 - [x]  IOULoss
 - [x]  OnlineHardExampleMiningCELoss
 - [x]  SSIMLoss
 - [x]  VggLoss 


<br>  

---
## 使用说明
- 配置文件通过python格式配置, 把损失函数的配置写到[配置](config.md)里
- 有两种配置方式 LossCompose 和 LossKeyCompose

---
### LossCompose  
- 这个函数的输入只能是一对 (fake, target)
- 所有的损失函数的输入形式必须一致
- 配置文件定义
    - name 损失函数名
    - 其他字段，损失函数的参数
    - 示例
    ```python
        loss_cfg = [
            dict(name='CrossEntropyLoss', weights=1.0),
            dict(name='MSELoss', weights=1.0),
        ]
    ```

- 构造损失函数
    - 通过 engine.loss.pipe 的 LossCompose
    - 示例
    ```python
    from engine.loss.pipe import LossCompose
    loss_func = LossCompose(loss_cfg)
    ```

- 使用
    - 通过 [model](model.md) 的 g_model 生成结果  
    ```python
    fake_data = g_model(input_data)
    loss = loss_func((fake_data, target_data))
    ```

---
### `LossKeyCompose`  [推荐]
- 配置文件定义
    - name 损失函数的名称
    - param 损失函数的参数
    - input_name 损失函数的forward方法的输入参数名, 如果所有的损失函数的forward方法的输入参数名一致就不用指定
    - 示例1  
    损失函数和(gt, fake) 一一对应
        ```python
        loss_cfg = dict(
            loss1=[
                dict(name='CrossEntropyLoss', 
                    param=dict(lambda_weight=1.0), 
                    input_name=['input_tensor1', 'target_tensor1']),
                dict(name='MSELoss', 
                    param=dict(lambda_weight=1.0), 
                    input_name=['input_tensor1', 'target_tensor1']),
            ]
        )

        loss_func = LossKeyCompose(loss_cfg)

        fake1 = g_model1(input_data)
        fake2 = g_model2(input_data)

        loss1_cfg = [
            """
            输入和loss1的长度一样，则表示每个输入和loss一一对应
            """
            dict(input_tensor1=fake1, target_tensor1=target1),
            dict(input_tensor1=fake2, target_tensor1=target2),
        ]


        total_loss = loss_func(dict(loss1=loss1_cfg))

        ```

    - 示例2
    多个损失函数对应一个(fake, gt)
        ```python
                loss_cfg = dict(
                    loss1=[
                        dict(name='CrossEntropyLoss', 
                            param=dict(lambda_weight=1.0), 
                            input_name=['input_tensor1', 'target_tensor1']),
                        dict(name='MSELoss', 
                            param=dict(lambda_weight=1.0), 
                            input_name=['input_tensor1', 'target_tensor1']),
                    ]
                )

                loss_func = LossKeyCompose(loss_cfg)

                fake1 = g_model1(input_data)

                loss1_cfg = [
                    dict(input_tensor1=fake1, target_tensor1=target1),
                ]

                total_loss = loss_func(dict(loss1=loss1_cfg))

        ```

    - 示例3  
    一个 loss 对应 多个 （fake, gt) 
        ```python
                loss_cfg = dict(
                    loss1=[
                        dict(name='CrossEntropyLoss', 
                            param=dict(lambda_weight=1.0), 
                            input_name=['input_tensor1', 'target_tensor1'])
                    ]
                )

                loss_func = LossKeyCompose(loss_cfg)

                fake1 = g_model1(input_data)
                fake2 = g_model2(input_data)

                loss1_cfg = [
                    """
                    输入和loss1的长度一样，则表示每个输入和loss一一对应
                    """
                    dict(input_tensor1=fake1, target_tensor1=target1),
                    dict(input_tensor1=fake2, target_tensor1=target2),
                ]

                total_loss = loss_func(dict(loss1=loss1_cfg))

            ```
    - 示例4 
        ```python
        loss_cfg = dict(
                # loss1 如果 (gt, fake) 和 loss1的长度一致，则表示 他们一一对应， 
                # loss1 如果 (gt, fake) 只有一个，则表示(gt, fake)对应所有的loss1，
                loss1=[
                    dict(name='MSELoss', 
                        param=dict(lambda_weight=1.0), 
                        input_name=['input_tensor1', 'target_tensor1']),
                    dict(name='MSELoss', 
                        param=dict(lambda_weight=1.0), 
                        input_name=['input_tensor2', 'target_tensor2']),
                    dict(name='VggLoss', 
                        param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
                    dict(name='VggLoss', 
                        param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
                ],
                # loss2 如果 (gt, fake) 和 loss2 他们一一对应， 
                # loss2 如果 (gt, fake) 有多个，则表示所有的(gt, fake)对应loss2，
                loss2=[
                    dict(name='MSELoss', 
                        param=dict(lambda_weight=1.0), 
                        input_name=['input_tensor1', 'target_tensor1']),  
                ]
                # （gt, fake) 的长度必须和 loss3 相同， (gt, fake)[0] 对应所有loss3[0]
                loss3=[
                    (  
                        # 在同一组下的loss的输入变量名保持一致; 如果变量名不一致，请保持输入个数一致， 顺序一致
                        dict(name='MSELoss', 
                            param=dict(param1=1.0), 
                            input_name=['input1', 'input2']),
                        dict(name='VggLoss', 
                            param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
                    ),
                    (
                        dict(name='MSELoss', 
                            param=dict(param1=1.0), 
                            input_name=['input1', 'input2']),
                        dict(name='VggLoss', 
                            param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
                    ),
                ]
            )
           
        ```  

<br>  

## 自定义损失函数
- 必须通过engine.loss.build.LOSS_ARCH_REGISTRY注册  

- 实现损失类  

```python
import torch
from engine.loss.build import LOSS_ARCH_REGISTRY


@LOSS_ARCH_REGISTRY.register()
class SSIMLoss(torch.nn.Module):
    def __init__(self, lambda_weight=1.):
        super(SSIMLoss, self).__init__()
        self.lambda_weight = lambda_weight
        # todo
        return

    def forward(self, x, target):
        # todo
        return self.lambda_weight * loss

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'lambda_weight: {})'.format(self.lambda_weight)
        return format_string

```

- 导入到程序 
- 通过[配置](#使用说明) 配置损失函数
