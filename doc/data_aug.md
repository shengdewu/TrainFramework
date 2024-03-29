# 数据增强方法介绍 [engine/transforms]

数据增强支持像素[图片，掩码等]、点[人脸点位，骨骼点位等]、框[人脸框，人体框等]的增强  

### ***像素增强***
 
----
#### `RandomBrightness`

- 描述:  
    随机调节图片的亮度

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    |brightness_limit| Union[float, Tuple[float]] | 亮度的程度范围， 如果是一个浮点数则取值范围是：[1-brightness_limit, 1+brightness_limit], 如果是元组则取值范围就是元组本身|
    | p | float | 生效的概率， 取值范围 [0, 1.0] |  

---  
<br>  

#### `RandomContrast`

- 描述:  
    随机调节图片的对比度  

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    | contrast_limit | Union[float, Tuple[float]] | 对比度程度范围， 如果是一个浮点数则取值范围是：[1-contrast_limit, 1+contrast_limit], 如果是元组则取值范围就是元组本身 |
    | p | float | 生效的概率， 取值范围 [0, 1.0] |  

---  
<br>  

#### `RandomSaturation`  

- 描述:  
    随机调节图片的饱和度 

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    | saturation_limit | Union[float, Tuple[float]] | 饱和度程度范围， 如果是一个浮点数则取值范围是：[1-saturation_limit, 1+saturation_limit], 如果是元组则取值范围就是元组本身 |
    | p | float | 生效的概率， 取值范围 [0, 1.0] |     

---      
<br>  

#### `RandomHue`  

- 描述:  
    随机调节图片的色度 

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    | hue_limit | Union[float, Tuple[float]] | 色度程度范围， 如果是一个浮点数则取值范围是：[-hue_limit, hue_limit], 如果是元组则取值范围就是元组本身 |
    | p | float | 生效的概率， 取值范围 [0, 1.0] |     

---    
<br>  

#### `RandomGamma`  

- 描述:  
    随机gamma调节 

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    | gamma_limit | Union[float, Tuple[float]] | 伽马值范围， 如果是一个浮点数则取值范围是：[-gamma_limit, gamma_limit], 如果是元组则取值范围就是元组本身 |
    | p | float | 生效的概率， 取值范围 [0, 1.0] |    

---  
<br>  

#### `RandomCLAHE`  

- 描述:  
    对图像使用 对比度受限的自适应直方图均衡化

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    | clip_limit | Union[float, Tuple[float]] | 对比度限制的阈值 如果是一个浮点数则取值范围是：[-clip_limit, clip_limit], 如果是元组则取值范围就是元组本身 |
    | tile_grid_size | Union[int, Tuple[int]] | 直方图均衡化的网格大小 如果是一个整数则值为 [tile_grid_size, tile_grid_size] 否则是元组本身
    | p | float | 生效的概率， 取值范围 [0, 1.0] | 

---  
<br>  

#### `RandomCompress`  

- 描述:  
    随机图像质量压缩， 从指定的图像质量值中随机选择一个来压缩图片质量

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    | quality_lower | int | 图片的最低质量 |
    | quality_upper | int | 图片的最高质量 |
    | quality_step | int | 图片的质量的步长 图片的质量集合 = range(start=quality_lower, stop=quality_upper, step=quality_step) |
    |compression_type | str | 图片类型 [jpg, webp]|
    | p | float | 生效的概率， 取值范围 [0, 1.0] |      

---  
<br>  

#### `RandomSharpen`  

- 描述:  

    随机锐化图像

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    | alpha | Union[float, Tuple[float]] | 锐化图片的可见范围 [0, 1.0] |
    | lightness | Union[float, Tuple[float]] | 锐化图片的亮度范围 |
    | p | float | 生效的概率， 取值范围 [0, 1.0] |   

---  
<br>  

#### `RandomToneCurve`  

- 描述:  
    通过连个正太分布[他们的标准方差相同=scale], 选取两个点操纵色调曲线来重新调整图像亮区和暗区之间的关系  

- parameters:  

    | name | type | description |
    |--- | --- | ---|
    | scale| float | 标准方差，用于对随机距离进行采样以移动修改图像曲线的两个控制点， 取值范围 [0, 1.0] |
    | p | float | 生效的概率， 取值范围 [0, 1.0]

---  
<br>  

#### `RandomBrightnessContrast`  

- 描述:  
    随机调节图片的亮度和对比度  

- parameters: 

    | name | type | description | 
    |--- | --- | ---| 
    |brightness_limit| Union[float, Tuple[float]] | 参见 [RandomBrightness](#randombrightness)|
    | contrast_limit | Union[float, Tuple[float]] | 参见 [RandomContrast](#randomcontrast) |
    | brightness_by_max | bool | True： 亮度的调节基础值是图像数据类型的最大值， False： 亮度的调节基础值是图像数据的均值|
    | p | float | 生效的概率， 取值范围 [0, 1.0] |

---  
<br>  

#### `RandomGaussianBlur`  

- 描述:  
    随机高斯模糊
- parameters:  

    | name | type | description|
    | --- | --- | --- |
    | blur_limit| Union[float, Tuple[float]] | 高斯半径 如果是一个浮点数则取值范围是：[-blur_limit, blur_limit], 如果是元组则取值范围就是元组本身|
    | sigma_limit | Union[float, Tuple[float]] | 高斯方差 如果是一个浮点数则取值范围是：[-sigma_limit, sigma_limit], 如果是元组则取值范围就是元组本身 |
    | p | float | 生效的概率， 取值范围 [0, 1.0] |  

---  
<br>  

#### `ToGray`  

- 描述:  
    图像灰度化
- parameters:  

    | name | type | description|
    | --- | --- | --- |    
    | p | float | 生效的概率， 取值范围 [0, 1.0] |      

---  
<br> 

#### `Normalize`  

- 描述:  
    使用指定的方差和均值归一化图像 y = (x - mean) / std 
- parameters:  

    | name | type | description|
    | --- | --- | --- |   
    | mean | Union[float, Tuple[float]] | 每个通道需要被减的均值，如果是float则所有的通道共享同一个均值|  
    | std | Union[float, Tuple[float]] | 每个通道需要被除的方差，如果是float则所有的通道共享同一个方差| 
    | max_pixel_value | float | 图像对应的数据类型的最大值 |

---  
<br>  

#### `RandomColorJitter`  

- 描述:  
    随机颜色扰动  
    在随机亮度、对比度、饱和度、色调、高斯模糊、伽马矫正，对比度受限的自适应直方图均衡中随机顺序并依次执行以上所有操作

- parameters:  

    | name | type | description|
    | --- | --- | --- |   
    | brightness_limit| Union[float, Tuple[float]] | 同 [RandomBrightness](#randombrightness) |    
    | brightness_p | float | RandomBrightness 的执行概率 范围[0, 1.0] |
    | contrast_limit| Union[float, Tuple[float]] | 同 [RandomContrast](#randomcontrast) |    
    | contrast_p | float | RandomContrast 的执行概率 范围[0, 1.0] |
    | saturation_limit| Union[float, Tuple[float]] | 同 [RandomSaturation](#randomsaturation) |    
    | saturation_p | float | RandomSaturation 的执行概率 范围[0, 1.0] |            
    | blur_limit| Union[float, Tuple[float]] | 同 [RandomGaussianBlur](#randomgaussianblur) |    
    | sigma_limit| Union[float, Tuple[float]] | 同 [RandomGaussianBlur](#randomgaussianblur) |              
    | blur_p | float | [RandomGaussianBlur 的执行概率 范围[0, 1.0] |
    | gamma_limit| Union[float, Tuple[float]] | 同 [RandomGamma](#randomgamma) |    
    | gamma_p | float | RandomGamma 的执行概率 范围[0, 1.0] |    
    | clahe_limit| Union[float, Tuple[float]] | 同 [RandomCLAHE](#randomclahe) |    
    | clahe_p | float | RandomCLAHE]的执行概率 范围[0, 1.0] |                       


<br>  

### ***空间增强***  

---  
<br>  

#### `RandomAffine`  

- 描述  
    实现图片、类图片[皮肤蒙版]、点和框的随机仿射变换，包括平移(translation)、缩放(scale)、旋转(rotation)和剪切(shear)
- parameters: 

    | name | type | description|
    | --- | --- | --- |  
    | rotate_degree_range | Union[float, Tuple[float], List[float]] | 旋转的角度， 可以是一个float， 或者一个元组， 后者指定一系列的角度 |
    |rotate_range| bool | 如果为True rotate_degree, 则rotate_degree_range 的长度只能是2， 角度的选择会通过 random.uniform, 否则是 random.choice | 
    |max_translate_ratio | float | 平移程度 范围[-max_translate_ratio, max_translate_ratio] |
    |scaling_ratio_range| Tuple[float] | 缩放(scale)的范围|
    |max_shear_degree | float | 剪切的范围 [-max_shear_degree, max_shear_degree]|
    | p | float | 生效的概率， 取值范围 [0, 1.0] | 
    |border_val| int | 三个通道的填充值 默认 114 |
    | clip_border | bool | 是否剪切掉在图像外面的物体 对点和box生效|
    | border_ratio | float | 仿射变换时图像的放大比率|
    | min_bbox_size| float | 变换之后的 bbox 的长或者框如果小于这个值将被移除
    | min_area_ratio| float| 原始的bbox与变换之后的 bbox 之间的面积比阈值 如果小于这个值将被移除|
    |max_aspect_ratio| float | 变换后的box的长和宽之间的比值 max(h/w, w/h) 大于这个阈值将被移除|

---  
<br>  

#### `RandomFlip`  

- 描述  
    实现图片、类图片[皮肤蒙版]、点和框的随机翻转
- parameters:  

    | name | type | description|
    | --- | --- | --- | 
    | direction | Union[str, List] | 翻转方向，取值必须是['horizontal', 'vertical', 'diagonal']其中的一个、几个或者全部| 
    | p | float | 生效的概率， 取值范围 [0, 1.0] | 

---  
<br>  

#### `Resize`  

- 描述  
    实现图片、类图片[皮肤蒙版]、点和框的resize
- parameters

    | name | type | description|
    | --- | --- | --- | 
    | target_size | int | 图片、类图片[皮肤蒙版]、点和框resize后的大小| 
    | interpolation | str | 插值类型  [INTER_AREA, INTER_LINEAR, INTEAR_NEAREST] |
    |keep_ratio| bool | 如果是true表示按照最长边缩放到target_size， 否则长和宽都缩放到target_size |
    |is_padding| bool | 只对keep_ratio=True生效，表示短边要填充到target_size |
    |clip_border| bool |  是否剪切掉在图像外面的物体 对点和box生效 |

---  
<br>  

#### `RandomResize`  

- 描述  
    随机缩放图片、类图片[皮肤蒙版]、点和框，再padding到指定大小
- parameters  

    | name | type | description|
    | --- | --- | --- | 
    |max_edge_length | Union[int, List] | 从max_edge_length随机选择一个尺寸来resize |
    |padding_size | int | 缩放后在填充后的大小| 
    |keep_ratio| bool | 如果是true表示按照最长边缩放到target_size， 否则长和宽都缩放到target_size |
    |clip_border| bool |  是否剪切掉在图像外面的物体 对点和box生效 |        

---     
<br>  

#### `RandomCrop`  

- 描述  
    随机裁剪图片、类图片[皮肤蒙版]、点和框
- parameters

    | name | type | description|
    | --- | --- | --- | 
    | min_crop_ratio | float ｜ 长宽的最小裁剪比率 取值范围 [0, 1.0] |
    | max_crop_ratio | float | 长宽的最大裁剪比率 取值范围 [0, 1.0] |   
    | crop_step | float | 裁剪比率的步长 |
    |clip_border| bool |  是否剪切掉在图像外面的物体 对点和box生效 |  
    | p | float | 生效的概率， 取值范围 [0, 1.0] |

---  
<br>  

#### `Pad32`  

- 描述  
    把增强内容pad到能被32整除  

 <br>  

# 数据增强的使用  

数据增强的使用是通过配置文件来初始化,可以是1个或多个  

### 定义配置文件
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
        )
    ]
```
### 初始化  

通过 TransformCompose 来初始化数据增强  

```python
from engine.transforms.pip import TransformCompose

transformers = TransformCompose(cfg)

```

### 使用  
- *必须使用如下格式* 返回值也是这个格式，不过会新增字段
    ```python
        aug_format = dict(
            color_fields=['img'],  
            img_fields=['img', 'mask'],
            pts_fields=['pts'],    
            bbox_fields=['bbox'], 
            pad_value=dict(img=0, mask=255, pts=0, bbox=0),
            img=np.ndarray, 
            mask=np.ndarray,
            pts=np.ndarray,  
            bbox=np.ndarray  
            img_shape=ori_img.shape[:2],
            ori_shape=ori_img.shape[:2]，
            pad_offset=[top, bottom, left, right],
            scale=[w, h]
        )
    ```  
  
    | name | type  | necessary | description |
    | --- | --- | --- | --- |
    |color_fields | List | yes | [像素增强](#像素增强)中必须字段<br> 这个字段里面的都会执行 [像素增强](#像素增强) | 
    |img_fields | List | no | [空间增强](#空间增强)中需要执行<big>图片</big>增强的字段， 如果指定了空间增强，则这个字段必须|
    |pts_fields| List | no | [空间增强](#空间增强)中需要执行<big>点位</big>增强的字段， 如果指定了空间增强，则这个字段必须 |
    |bbox_fields | List | no | [空间增强](#空间增强)中需要执行<big>框</big>增强的字段， 如果指定了空间增强，则这个字段必须 |  
    |pad_value | Dict | no | 对应字段填充值，<br>比如[Resize](#resize) 如果指定了则使用指定的值，否则使用内部指定值 |  
    |pad_offset | List | no | 经过[空间增强](#空间增强)后表示数据的填充大小|
    |scale| Tuple | no | 经过[空间增强](#空间增强)后表示数据相对于原始数据的在w和h上的缩放尺度|
    |img_shape | Tuple | no | 增强后的图片长宽 [height, width] |
    |ori_shape | Tuple | no | 原始的图片长宽 [height, width] |
    | img | np.ndarry | no | 表示 color_fields 和 img_fields 需要的字段, 这个名字用户可以自定义， *增强后的内容会放在这个字段覆盖原来的内容*|
    | mask | np.ndarry | no | 同 img *增强后的内容会放在这个字段覆盖原来的内容*|
    | pts | np.ndarry | no | 表示 pts_fields 需要的字段 *增强后的内容会放在这个字段覆盖原来的内容*|
    | bbox | np.ndarry | no | 表示 bbox_fields 需要的字段 *增强后的内容会放在这个字段覆盖原来的内容*|

- 根据上述说明填好需要的内容后调用方法  

    ```python
    aug_format = transformer(aug_format)
    ```
- 根据上述字段说明取出需要的内容  

<br>  

# 自定义用户的数据增强  

---
## 像素增强  

- 继承 BasicColorTransform, 实现 apply 方法， 详情如下:  

    class BasicColorTransform  
    ***
    - 方法  
        1. def __init__(self, p: float = 1.0)
            - 构造函数
            - 参数:  
                - p 这个方法的执行概率， 由子类传入
        2. def __call__(self, results)
            - [子类不用重载]<br>
              内部会循环调用 apply 方法，为 [输入参数](#使用) 中的 color_fields 每个字段执行数据增强操作
            - 参数:  
                #### Results 方法的[参数格式](#使用)
        3. def apply(self, img: np.ndarray, **params)
            - 由子类实现[必须]<br>
              数据增强的实现接口，如果 [输入参数](#使用) 中的 color_fields 有多个，则这个方法会多次调用
            - 参数:  
                - img 需要执行随机增强的字段，就是[输入参数](#使用) 中 color_fields 指定的字段中的一个

        3. def get_params(self, results) -> Dict[str, Any]
            - 由子类实现, [不是必须]<br> 
              在 __call__ 中 循环调用 apply 执行增强前，会先执行 这个方法， 为[输入参数](#使用) 中的 color_fields 每个字段确定相同的参数<br>
              比如在[随机亮度](#randombrightness)中，在[输入参数](#使用) 中 color_fields 有多个， 这个时候就要保证这一轮增强时[输入参数](#使用) 中 color_fields 他们的增强亮度一致

- 注册 engine.transforms.build 中的 BUILD_TRANSFORMER_REGISTRY， 详情如下:  

```python
from engine.transforms.build import BUILD_TRANSFORMER_REGISTRY

@BUILD_TRANSFORMER_REGISTRY.register()
class CustomAug(BasicColorTransform):
    def __init__(self, param=(0.2, 0.5), p=0.5):
        super(CustomAug, self).__init__(p, )
        self.param = param
        return

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.adjust_brightness(img, params['factor'])

    def get_params(self, kwargs):
        return {
            "factor": random.uniform(self.param[0], self.param[1]),
        }

```
- 导入 通过`import`导入到程序中

- 使用 参见[数据增强的使用](#数据增强的使用)  

---  
<br>  

## 空间增强

- 实现的类里面必须要针对[数据输入格式](#使用)中的 img_fields，pts_fields， bbox_fields中的字段实现对应的方法， 参考[随机翻转](#randomflip)  

- 注册 参考[像素增强](#像素增强)
- 导入 参考[像素增强](#像素增强)
- 使用 参见[数据增强的使用](#数据增强的使用)  
