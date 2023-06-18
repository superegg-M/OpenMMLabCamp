### PSPNet

- 配置文件：[pspnet-watermelon87.py](mmsegmentation%2Fpspnet-watermelon87.py)
- 训练日志：[20230617_173432](work_dirs%2F20230617_173432)
- 验证集评估日志：[20230618_111957](work_dirs%2F20230618_111957)
- 验证集评估指标：
```
06/18 11:20:04 - mmengine - INFO - Load checkpoint from work_dirs/iter_3000.pth
06/18 11:20:18 - mmengine - INFO - per class results:
06/18 11:20:18 - mmengine - INFO - 
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
|    red     | 87.05 | 95.37 |
|   green    | 89.94 | 93.74 |
|   white    | 48.12 | 63.15 |
| seed_black | 66.27 | 69.37 |
| seed-white | 62.58 | 74.98 |
| Unlabeled  |  6.1  |  6.1  |
+------------+-------+-------+
06/18 11:20:18 - mmengine - INFO - Iter(test) [11/11]    aAcc: 90.6700  mIoU: 60.0100  mAcc: 67.1200  data_time: 0.0160  time: 1.2851
```

预测图：

![watermelon_pspnet.jpg](outputs%2Fwatermelon_pspnet.jpg)

预测视频：

![watermelon_pspnet.gif](watermelon_pspnet.gif)
