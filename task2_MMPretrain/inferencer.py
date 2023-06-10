from mmpretrain import ImageClassificationInferencer

imagelist = ['草莓.jpeg','菠萝.jpeg']
inferencer = ImageClassificationInferencer('config/resnet50_fruit30.py', pretrained='work_dirs/epoch_10.pth')
results = inferencer(imagelist, show=True)

print(results[0]['pred_class'])
print(results[1]['pred_class'])