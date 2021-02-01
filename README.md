#### 环境

python3.6 ,  pytorch1.5.0，cuda9/9.2/10

#### 数据集

VOC2012数据集，VOC增强数据集（VOC2012+SBD）

```python
training_data = Voc2012('/home/gb502/wcl/VOC2012', 'train',transform=train_transforms)
## VOC增强数据集(VOC2012+SBD)
#training_data = Voc2012('/home/gb502/wcl/VOC2012', 'train',transform=train_transforms)
test_data = Voc2012('/home/gb502/wcl/VOC2012', 'val',transform=test_transforms)
```

#### 运行：

```shell
## shell运行
sh test.sh
##或者 python运行 裁剪尺寸为321×321
python train.py --bata53 0.3 --bata43 0.7 --bata52 0.2 --bata42 0.3 --bata32 0.5 --in_size_h 321 --in_size_w 321 --test_size_h 321 --test_size_w 321 --batch_size 8 --epochs 100
##或者 python运行 裁剪尺寸为512×512
#python train.py --bata53 0.3 --bata43 0.7 --bata52 0.2 --bata42 0.3 --bata32 0.5 --in_size_h 512 --in_size_w 512 --test_size_h 321 --test_size_w 321 --batch_size 8 --epochs 100 
```

#### 结果：

```shell
./log/batch_size8bata530.3bata520.2bata430.7bata420.3bata320.5_512,512_512,512_cfab_ratio_8.0_gau_mul_guid_weight_2021-1-27_9-37-44
```

| 方法  | 基础网络               | mIoU(%) | PA(%) |
| ----- | ---------------------- | ------- | ----- |
| PANet | 带膨胀卷积的ResNet-101 | 79.38   | 95.25 |
| our   | ResNet-101             | 80.28   | 92.64 |

