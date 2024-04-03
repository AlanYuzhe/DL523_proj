#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# 假设CelebA图像存储在"CelebA/CelebA/Img/img_align_celeba"，标注在"CelebA/CelebA/Anno/list_attr_celeba.txt"

def load_celeba_images_and_labels(image_dir, label_file):
    # 加载和预处理图像
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    images = np.array([np.array(Image.open(path).resize((64, 64))) for path in image_paths])

    # 加载标签（需要根据您的需求进行调整）
    with open(label_file, 'r') as file:
        labels = file.readlines()
    
    return images, labels

# 加载数据
x_train, y_train = load_celeba_images_and_labels('CelebA/CelebA/Img/img_align_celeba', 'CelebA/CelebA/Anno/list_attr_celeba.txt')
x_test, y_test = x_train, y_train  # 仅作示例；根据需要分割数据

# 归一化和转换
x_train = (x_train / 255.0).astype(np.float32)
x_test = (x_test / 255.0).astype(np.float32)
def mask(X, coords):
    """
    根据给定坐标修改蒙版区域，确保遮盖正确的图像部分。
    X: 图像数组。
    coords: 遮盖区域的坐标，(x0, y0, x1, y1)。
    """
    x0, y0, x1, y1 = coords
    X[:, y0:y1, x0:x1, :] = 0  # 应用遮盖
    return X

# 仅遮盖一个小区域，例如图像中心的10x10区域
masked_x_train = mask(np.copy(x_train), (27, 27, 37, 37))  # 调整坐标以遮盖中心小区域
masked_x_test = mask(np.copy(x_test), (27, 27, 37, 37))  # 同上

# 以下部分保持不变
model_path = './my_model.h5'
model = load_model(model_path)
print("Model loaded successfully.")

# 由于内存或GPU资源问题，建议在CPU上评估
with tf.device('/CPU:0'):
    test_loss = model.evaluate(masked_x_test, x_test)  # 注意：这里使用x_test作为y_test的占位符
    print('Test loss:', test_loss)

# 选择测试集中的一个图像进行预测和显示
idx = np.random.randint(0, len(x_test))
test_img = x_test[idx]
masked_img = masked_x_test[idx]  # 使用遮盖后的图像

# 使用模型进行预测
gen_img = model.predict(np.expand_dims(masked_img, axis=0))[0]

# 绘制被遮盖的输入图像、生成的输出和原始图像
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(masked_img)
plt.title('Masked Input')
plt.subplot(1, 3, 2)
plt.imshow(gen_img)
plt.title('Generated Output')
plt.subplot(1, 3, 3)
plt.imshow(test_img)
plt.title('Original Image')
plt.show()

