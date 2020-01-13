import os 
import glob
import numpy as np 
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.utils import to_categorical, np_utils

train_path = "../Resources/train"
test_path = "../Resources/test"
names = []
names_map = {}
num_classes = None

def map_folder_names(folder_names):
    global names, names_map, num_classes
    names = folder_names
    names_map = {name: i for i,name in enumerate(names)}
    num_classes = len(names)

def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode_label(labels):
    "convert class labels to one-hot encoding"
    global num_classes
    return np_utils.to_categorical(labels, num_classes)

def get_images(train_or_test_path, folder_names):
    img_data_list=[]
    labels = []
    map_folder_names(folder_names)
    for dataset in os.listdir(train_or_test_path):
        if dataset[0] == ".": continue
        print(train_or_test_path + "/" + dataset + "/*")
        img_list = glob.glob(train_or_test_path + "/" + dataset + "/*")
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            img_path = img
            if ".DS_Store" in img_path: continue
            img = process_image(img_path)
            img_data_list.append(img)
            labels.append(names_map[dataset.lower()])
    img_data = np.array(img_data_list)
    img_data=np.rollaxis(img_data,1,0)
    img_data=img_data[0]
    X = img_data
    Y = encode_label(labels)
    return (X, Y)

# X, Y = get_images(train_path)