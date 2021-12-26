import os
from shutil import copyfile
import pandas as pd
import random

ROOT_DIR = '/gscratch/ubicomp/hughsun/HAM10000/ResNet'
split_size = 0.8

def getMergedHAMDataset():
    '''
    unzipping the HAM10000.zip gives 2 folders of image data. Each
    folder contains about 5000 images. This function produces a single folder
    named HAM10000 containing all the images.
    '''
    part1_imgs = os.listdir(f'{ROOT_DIR}/HAM10000_images_part_1')
    for img in part1_imgs:
        copyfile(os.path.join(f'{ROOT_DIR}/HAM10000_images_part_1', img), os.path.join(f'{ROOT_DIR}/HAM10000_images_part_2', img))
    os.rename('HAM10000_images_part_2', 'HAM10000')

def createDataFolders():
    '''
    create the necessary image folders for training/validating.
    '''
    os.mkdir(f'{ROOT_DIR}/orig/')
    os.mkdir(f'{ROOT_DIR}/orig/0.MEL/')
    os.mkdir(f'{ROOT_DIR}/orig/1.NV/')
    os.mkdir(f'{ROOT_DIR}/orig/2.BCC/')
    os.mkdir(f'{ROOT_DIR}/orig/3.AKIEC/')
    os.mkdir(f'{ROOT_DIR}/orig/4.BKL/')
    os.mkdir(f'{ROOT_DIR}/orig/5.DF/')
    os.mkdir(f'{ROOT_DIR}/orig/6.VASC/')

    os.mkdir(f'{ROOT_DIR}/skin/')
    os.mkdir(f'{ROOT_DIR}/skin/training/')
    os.mkdir(f'{ROOT_DIR}/skin/validation/')
    os.mkdir(f'{ROOT_DIR}/skin/training/0.MEL/')
    os.mkdir(f'{ROOT_DIR}/skin/training/1.NV/')
    os.mkdir(f'{ROOT_DIR}/skin/training/2.BCC/')
    os.mkdir(f'{ROOT_DIR}/skin/training/3.AKIEC/')
    os.mkdir(f'{ROOT_DIR}/skin/training/4.BKL/')
    os.mkdir(f'{ROOT_DIR}/skin/training/5.DF/')
    os.mkdir(f'{ROOT_DIR}/skin/training/6.VASC/')
    os.mkdir(f'{ROOT_DIR}/skin/validation/0.MEL/')
    os.mkdir(f'{ROOT_DIR}/skin/validation/1.NV/')
    os.mkdir(f'{ROOT_DIR}/skin/validation/2.BCC/')
    os.mkdir(f'{ROOT_DIR}/skin/validation/3.AKIEC/')
    os.mkdir(f'{ROOT_DIR}/skin/validation/4.BKL/')
    os.mkdir(f'{ROOT_DIR}/skin/validation/5.DF/')
    os.mkdir(f'{ROOT_DIR}/skin/validation/6.VASC/')

def split_data(source, training, validation, split_size):
    random.seed(1234) # set the seed so we have reproducible train/valid splits.

    files = []
    for filename in os.listdir(source):
        file = source + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * split_size)
    validation_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    validation_set = shuffled_set[:validation_length]

    for filename in training_set:
        this_file = source + filename
        destination = training + filename
        copyfile(this_file, destination)

    for filename in validation_set:
        this_file = source + filename
        destination = validation + filename
        copyfile(this_file, destination)

def generate_train_val_folders():
    '''
    generate folders for train and validation separately. Copy necessary
    data into each folder.
    '''
    orig_all = f'{ROOT_DIR}/HAM10000/'
    source_MEL = f'{ROOT_DIR}/orig/0.MEL/'
    source_NV = f'{ROOT_DIR}/orig/1.NV/'
    source_BCC = f'{ROOT_DIR}/orig/2.BCC/'
    source_AKIEC = f'{ROOT_DIR}/orig/3.AKIEC/'
    source_BKL = f'{ROOT_DIR}/orig/4.BKL/'
    source_DF = f'{ROOT_DIR}/orig/5.DF/'
    source_VASC = f'{ROOT_DIR}/orig/6.VASC/'

    training_MEL = f'{ROOT_DIR}/skin/training/0.MEL/'
    training_NV = f'{ROOT_DIR}/skin/training/1.NV/'
    training_BCC = f'{ROOT_DIR}/skin/training/2.BCC/'
    training_AKIEC = f'{ROOT_DIR}/skin/training/3.AKIEC/'
    training_BKL = f'{ROOT_DIR}/skin/training/4.BKL/'
    training_DF = f'{ROOT_DIR}/skin/training/5.DF/'
    training_VASC = f'{ROOT_DIR}/skin/training/6.VASC/'
    validation_MEL = f'{ROOT_DIR}/skin/validation/0.MEL/'
    validation_NV = f'{ROOT_DIR}/skin/validation/1.NV/'
    validation_BCC = f'{ROOT_DIR}/skin/validation/2.BCC/'
    validation_AKIEC = f'{ROOT_DIR}/skin/validation/3.AKIEC/'
    validation_BKL = f'{ROOT_DIR}/skin/validation/4.BKL/'
    validation_DF = f'{ROOT_DIR}/skin/validation/5.DF/'
    validation_VASC = f'{ROOT_DIR}/skin/validation/6.VASC/'

    data = pd.read_csv(f'{ROOT_DIR}/HAM10000_metadata.csv')

    data = data.drop_duplicates(subset='lesion_id')

    """##
    ###  Now using list comprehensions we will save all image names and their diagnosis in variables, and using for loop we will sort and copy all files by their diagnosis into the corresponding folders.
    ### Then we will split each individual class of photos into train and val datasets 80/20%.
    """

    image_names = [x for x in data['image_id']]
    diagnosis = [x for x in data['dx']]

    for index, image in enumerate(image_names):
        image = image + '.jpg'
        if diagnosis[index] == 'mel':
            copyfile(os.path.join(orig_all, image), os.path.join(source_MEL, image))
        elif diagnosis[index] == 'nv':
            copyfile(os.path.join(orig_all, image), os.path.join(source_NV, image))
        elif diagnosis[index] == 'bcc': 
            copyfile(os.path.join(orig_all, image), os.path.join(source_BCC, image))
        elif diagnosis[index] == 'akiec': 
            copyfile(os.path.join(orig_all, image), os.path.join(source_AKIEC, image))
        elif diagnosis[index] == 'bkl':
            copyfile(os.path.join(orig_all, image), os.path.join(source_BKL, image))
        elif diagnosis[index] == 'df':  
            copyfile(os.path.join(orig_all, image), os.path.join(source_DF, image))
        elif diagnosis[index] == 'vasc': 
            copyfile(os.path.join(orig_all, image), os.path.join(source_VASC, image))

    split_data(source_MEL, training_MEL, validation_MEL, split_size)
    split_data(source_NV, training_NV, validation_NV, split_size)
    split_data(source_BCC, training_BCC, validation_BCC, split_size)
    split_data(source_AKIEC, training_AKIEC, validation_AKIEC, split_size)
    split_data(source_BKL, training_BKL, validation_BKL, split_size)
    split_data(source_DF, training_DF, validation_DF, split_size)
    split_data(source_VASC, training_VASC, validation_VASC, split_size)

def main():
    # only execute the following two functions once.
    getMergedHAMDataset()
    createDataFolders()
    
    generate_train_val_folders()
    print('----data train test split complete----')

main()