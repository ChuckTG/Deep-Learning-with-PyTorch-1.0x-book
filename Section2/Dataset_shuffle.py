import torch
import glob
import numpy as np
import os

path = 'Data/Train_Data'

dog_files = [f for f in glob.glob('Data/Train_Data/dog/*.jpg') ]
cat_files = [f for f in glob.glob('Data/Train_Data/cat/*.jpg') ]

files = dog_files + cat_files
print(f"Total number of images {len(files)}")
n_images = len(files)

shuffle = np.random.permutation(n_images)

os.mkdir(os.path.join(path,'train'))
os.mkdir(os.path.join(path,'valid'))
for t in ['train','valid'] :
    for folder in ['dog/','cat/']:
        os.mkdir(os.path.join(path,t,folder))

#create random train data
for i in shuffle[:250]:
    folder = files[i].split('/')[-2].split('.')[0]
    image  = files[i].split('/')[-1]
    os.rename(files[i],os.path.join(path,'train',folder,image))

#create random validation data
for i in shuffle[250:]:
    folder = files[i].split('/')[-2].split('.')[0]
    image = files[i].split('/')[-1]
    os.rename(files[i], os.path.join(path, 'valid', folder, image))

