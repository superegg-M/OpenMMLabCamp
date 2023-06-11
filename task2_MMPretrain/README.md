### ResNet50（epoch=10）
- 配置文件：[resnet50_fruit30.py](config%2Fresnet50_fruit30.py)
- 训练日志：[20230611_081547](work_dirs%2F20230611_081547)
- 验证集评估日志：[20230611_171152](work_dirs%2Fresnet50_fruit30%2F20230611_171152)
- 验证集评估指标：
```
06/11 17:11:59 - mmengine - INFO - Load checkpoint from work_dirs/epoch_10.pth
06/11 17:12:02 - mmengine - INFO - Epoch(test) [10/28]    eta: 0:00:05  time: 0.3044  data_time: 0.0241  memory: 452  
06/11 17:12:03 - mmengine - INFO - Epoch(test) [20/28]    eta: 0:00:01  time: 0.0344  data_time: 0.0008  memory: 452  
06/11 17:12:03 - mmengine - INFO - Epoch(test) [28/28]    accuracy/top1: 84.7973  accuracy/top5: 97.5225  data_time: 0.0090  time: 0.1315
```
- 预测图片结果：

![草莓.png](output%2F%E8%8D%89%E8%8E%93.png)
![菠萝.png](output%2F%E8%8F%A0%E8%90%9D.png)