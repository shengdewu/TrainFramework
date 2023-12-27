#  损失函数说明

## LossCompose  

#### 调用形式
```python
    # 初始化
    loss_cfg = [
        dict(name='CrossEntropyLoss'),
        dict(name='MSELoss'),
    ]

    from engine.loss.pipe import LossCompose
    loss_func = LossCompose(loss_cfg)
    ...
    fake_data = g_model(input_data)
    loss = loss_func((fake_data, target_data))
```
#### 说明 
1.  这种配置下所有的损失函数的输入只能一同一种形式
2.  输入顺序必须和损失函数要求的的输入顺序一致

## LossKeyCompose  

### 说明
+ 如果 输入和loss_function的长度一样，表示输入和loss_function一一对应
+ 如果要表示 一组输入对应多个 loss_function 请使用 案例 D

##### A 每个key下的损失函数可以通过 input_name 指定对应的输入
```python
    # name: 损失函数的名称，必须注册
    # param: 损失函数的初始化参数
    # input_name: 损失函数的输入变量名
    loss_cfg = dict(
        loss1=[
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor1', 'target_tensor1']),
            dict(name='MSELoss', param=dict(lambda_weight=1.0), input_name=['input_tensor1', 'target_tensor1']),
        ],
        loss2=[
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor1', 'target_tensor1']),
            dict(name='MSELoss', param=dict(lambda_weight=1.0), input_name=['input_tensor1', 'target_tensor1']),
        ]
    )
    
    loss_func = LossKeyCompose(loss_cfg)

    ...

    fake1 = g_model1(input_data)
    fake2 = g_model2(input_data)

    loss1_cfg = [
        """
        输入和loss1的长度一样，则表示每个输入和loss一一对应
        """
        dict(input_tensor1=fake1, target_tensor1=target1),
        dict(input_tensor1=fake1, target_tensor1=target1),
    ]
    loss2_cfg = [
        dict(input_tensor1=fake2, target_tensor1=target2),
        dict(input_tensor1=fake2, target_tensor1=target2),
    ]
    total_loss = loss_func(dict(loss1=loss1_cfg, loss2=loss2_cfg))
   
```

##### B 每个key也可以不用指定输入名，则输入必须和损失函数要求的顺序一致  

```python
    loss_cfg = dict(
        loss1=[
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=10.)),
        ],
        loss2=[
            dict(name='MSELoss', param=dict(lambda_weight=10.))
        ]
    )
    
    loss_func = LossKeyCompose(loss_cfg)

    ...

    fake1 = g_model1(input_data)
    fake2 = g_model2(input_data)

    loss1_cfg = ((fake1, target1),)
    loss2_cfg = ((fake2, target2),)
    total_loss = loss_cfg(dict(loss1=loss1_cfg, loss2=loss2_cfg))
```  
##### C 每个key下的损失函数的输入可以不同,这种情况下建议使用 input_name指出
```python
   loss_cfg = dict(
        loss1=[
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor1', 'target_tensor1']),
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor2', 'target_tensor2']),
        ],
        loss2=[
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor3', 'target_tensor3']),
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor4', 'target_tensor4']),
        ]
    )
    
    loss_func = LossKeyCompose(loss_cfg)

    ...

    fake1 = g_model1(input_data)
    fake2 = g_model2(input_data)
    fake3 = g_model3(input_data)
    fake4 = g_model4(input_data)

    loss1_cfg = [
        dict(input_tensor1=fake1, target_tensor1=target1),
        dict(input_tensor2=fake2, target_tensor2=target2),
    ]
    loss2_cfg = [
        dict(input_tensor3=fake3, target_tensor3=target3),
        dict(input_tensor4=fake4, target_tensor4=target4),
    ]
    total_loss = loss_func(dict(loss1=loss1_cfg, loss2=loss2_cfg))
```

##### D 每个key下的一个子项  是 tuple或者 list eg: loss2, 表示一个输入或多个输入对应同一组loss_function
 ```python
   loss_cfg = dict(
        # A 同一个输入共享 loss下的所有loss
        loss1=[
            dict(name='MSELoss', param=dict(lambda_weight=1.0), input_name=['input_tensor1', 'target_tensor1']),
            dict(name='MSELoss', param=dict(lambda_weight=1.0), input_name=['input_tensor2', 'target_tensor2']),
            dict(name='VggLoss', param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
            dict(name='VggLoss', param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
        ],
        # B 同一个输入共享 loss2[0]下的所有loss
        loss2=[
            (  
                # 在同一组下的loss的输入变量名保持一致; 如果变量名不一致，请保持输入个数一致， 顺序一致
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2']),
                dict(name='VggLoss', param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
            ),
            (
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2']),
                dict(name='VggLoss', param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
            ),
            (
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2']),
                dict(name='VggLoss', param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
            ),
            (
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2']),
                dict(name='VggLoss', param=dict(lambda_weight=0.1, vgg_arch='vgg16', vgg_path='/mnt/sda1/workspace/retouch/AutoRetouch/vgg/vgg16-397923af.pth')),
            ),
        ]
    )
 
    """
    A 和 B 的区别， 对于单个输入 没有差别，当输入是多个的时候 eg 输入个数 == 4， B中每一个输入对应了2个loss
    """
    loss_func = LossKeyCompose(loss_cfg)

    # fakes 多个输出， gts 多个 gt
    loss_input = list()
    for fake, gt in zip(fakes, gts):
        loss_input.append((fake, gt))
    
    # loss2: 中每一个输入对应了2个loss， loss1: 中每一个输入对应了1个loss
    total_loss = loss_func(dict(loss1=loss_input, loss2=loss_input))

    
```