### RTMDet（epoch=40）
- 配置文件：[rtmdet_balloon.py](config%2Frtmdet_balloon.py)
- 训练日志：[20230611_221534](work_dirs%2Frtmdet_balloon%2F20230611_221534)
- 验证集评估日志：[20230611_223939](work_dirs%2Frtmdet_balloon%2F20230611_223939)
- 验证集评估指标：
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.761
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.856
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.845
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.899
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.228
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.780
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.842
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.708
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.933
06/11 22:39:57 - mmengine - INFO - bbox_mAP_copypaste: 0.761 0.856 0.845 0.000 0.442 0.899
06/11 22:39:58 - mmengine - INFO - Epoch(test) [13/13]    coco/bbox_mAP: 0.7610  coco/bbox_mAP_50: 0.8560  coco/bbox_mAP_75: 0.8450  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.4420  coco/bbox_mAP_l: 0.8990  data_time: 0.3845  time: 0.6938
```
- 预测图片结果：

![气球1.jpg](outputs%2Fvis%2F%E6%B0%94%E7%90%831.jpg)
![气球2.jpeg](outputs%2Fvis%2F%E6%B0%94%E7%90%832.jpeg)

- 可视化分析：

![气球1_neck.jpg](vis_output%2F%E6%B0%94%E7%90%831_neck.jpg)
backbone
![气球1_backbone.jpg](vis_output%2F%E6%B0%94%E7%90%831_backbone.jpg)
neck
![气球1_CAM_m.jpg](vis_output%2F%E6%B0%94%E7%90%831_CAM_m.jpg)
Box AM 

更多详见：[vis_output](vis_output)