import os
import shutil
from sklearn.model_selection import train_test_split

# 数据集路径
dataset_dir = "fruit30_train"
train_dir = "data/training_set"
val_dir = "data/val_set"

# 创建训练集和验证集文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 汉译英字典
ch2En={
    '黄瓜': 'cucumber',
    '香蕉': 'banana',
    '车厘子': 'cherry',
    '西红柿': 'tomato',
    '西瓜': 'watermelon',
    '葡萄-红': 'grape',
    '葡萄-白': 'white grape',
    '菠萝': 'pineapple',
    '荔枝': 'litchi',
    '草莓': 'strawberry',
    '苹果-青': 'green apple',
    '苹果-红': 'red apple',
    '苦瓜': 'bitter gourd',
    '芒果': 'mango',
    '脐橙': 'navel orange',
    '胡萝卜': 'carrot',
    '砂糖橘': 'Sugar orange',
    '石榴': 'pomegranate',
    '猕猴桃': 'kiwi fruit',
    '火龙果': 'dragon fruit',
    '榴莲': 'durian',
    '椰子': 'coconut',
    '梨': 'pear',
    '桂圆': 'longan',
    '柠檬': 'lemon',
    '柚子': 'pomelo',
    '杨梅': 'waxberry',
    '山竹': 'mangosteen',
    '圣女果': 'cherry tomato',
    '哈密瓜': 'cantaloupe'
}

# 遍历数据集文件夹，按类别划分图像到训练集和验证集文件夹
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # 创建类别文件夹
        train_class_dir = os.path.join(train_dir, ch2En[class_name])
        val_class_dir = os.path.join(val_dir, ch2En[class_name])
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # 获取类别图像列表
        image_list = os.listdir(class_dir)
        # 划分训练集和验证集
        train_images, val_images = train_test_split(image_list, test_size=0.2, random_state=42)

        # 将图像复制到训练集文件夹
        for image_name in train_images:
            src_path = os.path.join(class_dir, image_name)
            dst_path = os.path.join(train_class_dir, image_name)
            shutil.copyfile(src_path, dst_path)

        # 将图像复制到验证集文件夹
        for image_name in val_images:
            src_path = os.path.join(class_dir, image_name)
            dst_path = os.path.join(val_class_dir, image_name)
            shutil.copyfile(src_path, dst_path)
