import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

data_path = '/dmc/ml_storage/machine_learning/Final_ML_CBS/'
img_array = []
for index, file in enumerate(os.listdir(data_path + 'data/train')):
    img = Image.open(data_path + 'data/train/' + file)
    img = img.resize((512, 512))
    img_array.append(np.array(img))
    if index % 1000 == 0:
        print(index)

img_array = np.array(img_array)
np.save(data_path + 'data/train_images.npy', img_array)
