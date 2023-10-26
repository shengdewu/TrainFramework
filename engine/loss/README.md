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
    loss_func = LossKeyCompose(loss_cfg)
    ...
    fake_data = g_model(input_data)
    loss = loss_func((fake_data, target_data))
```
#### 说明 
1.  这种配置下所有的损失函数的输入只能一同一种形式
2.  输入顺序必须和损失函数要求的的输入顺序一致

## LossKeyCompose  

#### 说明
1. 每个key下的损失函数可以通过 input_name 指定对应的输入
```python
    # name: 损失函数的名称，必须注册
    # param: 损失函数的初始化参数
    # input_name: 损失函数的输入变量名
    loss_cfg = dict(
        loss1=[
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor1', 'target_tensor1']),
        ],
        loss2=[
            dict(name='CrossEntropyLoss', param=dict(lambda_weight=1.0), input_name=['input_tensor1', 'target_tensor1']),
        ]
    )
    
    loss_func = LossKeyCompose(loss_cfg)

    ...

    fake1 = g_model1(input_data)
    fake2 = g_model2(input_data)

    loss1_cfg = [
        dict(input_tensor1=fake1, target_tensor1=target1),
    ]
    loss2_cfg = [
        dict(input_tensor1=fake2, target_tensor1=target2),
    ]
    total_loss = loss_func(dict(loss1=loss1_cfg, loss2=loss2_cfg))
   
```  

2. 每个key也可以不用指定输入名，则输入必须和损失函数要求的顺序一致
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

    loss1_cfg = (fake1, target1)
    loss2_cfg = (fake2, target2)
    total_loss = loss_cfg(dict(loss1=loss1_cfg, loss2=loss2_cfg))
```  
3. 每个key下的损失函数的输入可以不同,这种情况下建议使用 input_name指出
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