import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

def load_celeba_images_and_labels(image_dir, label_file):
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    images = np.array([np.array(Image.open(path).resize((64, 64))) for path in image_paths])
    with open(label_file, 'r') as file:
        labels = file.readlines()
    
    return images, labels
x_train, y_train = load_celeba_images_and_labels('CelebA/CelebA/Img/img_align_celeba', 'CelebA/CelebA/Anno/list_attr_celeba.txt')
x_test, y_test = x_train, y_train 
x_train = (x_train / 255.0).astype(np.float32)
x_test = (x_test / 255.0).astype(np.float32)
def mask(X, coords):
    x0, y0, x1, y1 = coords
    X[:, y0:y1, x0:x1, :] = 0 
    return X
masked_x_train = mask(np.copy(x_train), (27, 27, 37, 37)) 
masked_x_test = mask(np.copy(x_test), (27, 27, 37, 37))  
model_path = './my_model.h5'
model = load_model(model_path)
print("Model loaded successfully.")

with tf.device('/CPU:0'):
    test_loss = model.evaluate(masked_x_test, x_test) 
    print('Test loss:', test_loss)

idx = np.random.randint(0, len(x_test))
test_img = x_test[idx]
masked_img = masked_x_test[idx]  
gen_img = model.predict(np.expand_dims(masked_img, axis=0))[0]
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

