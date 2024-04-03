import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, metrics
from tensorflow.keras.datasets import cifar10
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Assuming CelebA images are stored in "CelebA/CelebA/Img/img_align_celeba" and 
# annotations in "CelebA/CelebA/Anno/list_attr_celeba.txt"

def load_celeba_images_and_labels(image_dir, label_file):
    # Load and preprocess images
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
    images = np.array([np.array(Image.open(path).resize((64, 64))) for path in image_paths])

    # Load labels (this will need to be adapted based on how you want to use the labels)
    # For demonstration, this just loads a list of filenames. You'll need to parse the actual labels.
    with open(label_file, 'r') as file:
        labels = file.readlines()
    
    # Convert labels to a more usable format here, e.g., to_categorical if you're using labels for classification

    return images, labels

# Load data
x_train, y_train = load_celeba_images_and_labels('CelebA/CelebA/Img/img_align_celeba', 'CelebA/CelebA/Anno/list_attr_celeba.txt')
x_test, y_test = x_train, y_train  # This is just an example; split your data accordingly

# Normalize and convert
x_train = (x_train / 255.0).astype(np.float32)
x_test = (x_test / 255.0).astype(np.float32)

# Now you can correctly print min and max values
print("image range is {}, {}".format(np.min(x_test), np.max(x_test)))
print("new image range is {}, {}".format(np.min(x_test), np.max(x_test)))
def mask(X, coords):
    x0, y0, x1, y1 = coords
    X[:, y0:y1, x0:x1, :] = 0
    return X


masked_x_train = mask(np.copy(x_train), (16, 2, 30, 30))  
masked_x_test = mask(np.copy(x_test), (16, 2, 30, 30))  
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

def create_cnn_model(input_shape=(64, 64, 3)):  
    inputs = Input(input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model
random_state = 42
test_size = 0.1

x_train, x_val, _, _ = train_test_split(x_train, x_train, test_size=test_size, random_state=random_state)
masked_x_train, masked_x_val, _, _ = train_test_split(masked_x_train, masked_x_train, test_size=test_size, random_state=random_state)

print("Sets shapes: ",x_train.shape,x_val.shape,x_test.shape)
print("Masked sets shapes: ",masked_x_train.shape,masked_x_val.shape,masked_x_test.shape)
image_shape = x_train[0].shape

model = create_cnn_model(image_shape)
model.summary()
batch_size=256
epochs = 30
model_save_path = 'model1/cnnmodel'

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

with tf.device('/CPU:0'):
    history = model.fit(masked_x_train, x_train, batch_size=batch_size, epochs=epochs, validation_data=(masked_x_val, x_val))

model_save_full_path = os.path.join(model_save_path, 'my_model.h5')  
model.save(model_save_full_path)

print(f'Model saved to {model_save_full_path}')
