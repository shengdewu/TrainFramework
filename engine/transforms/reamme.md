## 数据增强使用说明
* 输入格式介绍 
```python
result = dict(
    color_fiedls=['img', 'mask],  #需要做颜色扰动的字段
    pts_fields=['pts'],    #需要做点位增强的字段
    bbox_fields=['bbox'],  #需要做框增强的字段
    img=np.ndarray,  #图像
    mask=np.ndarray, #掩码
    pts=np.ndarray,  #点位 [[x,y], ...]
    bbox=np.ndarray  #框 [[x1, y1, x2, y2], ...]
)
```
* 使用说明
```python
    cfg = [
        dict(
            name='RandomAffine',
            max_rotate_degree=30,
            border_ratio=1.2,
            p=1.0,
        ),
        dict(
            name='RandomFlip',
            direction=['horizontal', 'vertical', 'diagonal'],
            p=0.1
        ),
        dict(
            name='RandomCrop',
            p=1.0,
            max_crop_ratio=0.25,
            min_crop_ratio=0.05,
            crop_step=0.05
        ),
        dict(
            name='Resize',
            target_size=img_size,
            interpolation='INTER_LINEAR',
            keep_ratio=False,
            clip_border=True,
        ),
        dict(
            name='RandomResize',
            max_edge_length=[256],
            padding_size=img_size,
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
            saturation_p=0.6,
            hue_limit=[-0.1, 0.1],
            hue_p=0.2,
            blur_limit=[3, 7],
            blur_p=0.1,
            gamma_limit=[0.3, 3.0],
            gamma_p=0.1,
        )
    ]
    transformers = TransformCompose(cfg)

    result = dict(
        color_fiedls=['img', 'mask],  #需要做颜色扰动的字段
        pts_fields=['pts'],    #需要做点位增强的字段
        bbox_fields=['bbox'],  #需要做框增强的字段
        img=np.ndarray,  #图像
        mask=np.ndarray, #掩码
        pts=np.ndarray,  #点位 [[x,y], ...]
        bbox=np.ndarray  #框 [[x1, y1, x2, y2], ...]
        img_shape=ori_img.shape[:2],
        ori_shape=ori_img.shape[:2]
    )

    result = transformer(result)

```