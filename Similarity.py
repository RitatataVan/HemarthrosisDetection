import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from scipy import spatial
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import layers, Model, models
from tensorflow.keras.applications.vgg16 import VGG16
import random
from glob import glob

from PIL import Image, ImageEnhance, ImageOps


def load_real_images_by_class(folder):
    images = []
    for filename in glob(folder+'/*'):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.png']]):
            img = image.load_img(filename, target_size=(224, 224))
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            if img is not None:
                images.append([img, filename])
    return images


def load_synthetic_images_by_class(folder):
    images = []
    for name in os.listdir(folder):
        sub_dir_path = folder + "/" + name
        if os.path.isdir(sub_dir_path):
            for filename in glob(sub_dir_path + "/*/*"):
                if any([filename.endswith(x) for x in [".jpeg", ".jpg", ".png"]]):
                    img = image.load_img(filename, target_size=(224, 224))
                    img = img_to_array(img)
                    img = img.reshape((1,) + img.shape)
                    if img is not None:
                        images.append([img, name, filename])

    return images


def create_model():
    # loading vgg16 model and using all the layers until the 2 to the last to use all the learned cnn layers
    vgg = VGG16(include_top=True)
    model2 = Model(vgg.input, vgg.layers[-2].output)
    return model2


def get_fea(imgs_arr):
    fea = np.zeros((len(imgs_arr), 4096))
    imgs_arr = np.squeeze(np.array(imgs_arr))
    for i in range(imgs_arr.shape[0]):
        fea[i] = model.predict(np.expand_dims(imgs_arr[i], axis=0))
    return fea


def cosine_similarity(input1, input2):
    similarity = 1 - spatial.distance.cosine(input1, input2)
    rounded_similarity = int((similarity * 10000)) / 10000.0
    return rounded_similarity


def filter_batch(data, batch="Batch1"):
    batch_img = []
    for i in data:
        if batch in i[1]:
            batch_img.append(i)
    return batch_img


def compare_same_list(data_list, csv_name, all_=False):
    similarities = []
    imgname_1 = []
    imgname_2 = []
    
    if not all_: # If we are comparing positive against positive, and negative against negative
        # Negative
        neg_data_feat = get_fea([item[0] for item in data_list[0]])
        for i in range(len(neg_data_feat)):
            for j in range(i+1, len(neg_data_feat)):
                d = cosine_similarity(neg_data_feat[i], neg_data_feat[j])
                similarities.append(d)
                imgname_1.append(data_list[0][i][1])
                imgname_2.append(data_list[0][j][1])
        
        # Positive
        pos_data_feat = get_fea([item[0] for item in data_list[1]])
        for i in range(len(pos_data_feat)):
            for j in range(i+1, len(pos_data_feat)):
                d = cosine_similarity(pos_data_feat[i], pos_data_feat[j])
                similarities.append(d)
                imgname_1.append(data_list[1][i][1])
                imgname_2.append(data_list[1][j][1])
    
    elif all_: # If we are not making a distinction between positive and negative
        data_feat = get_fea([item[0] for item in data_list])
        
        for i in range(len(data_feat)):
            for j in range(i+1, len(data_feat)):
                d = cosine_similarity(data_feat[i], data_feat[j])
                similarities.append(d)
                imgname_1.append(data_list[i][1])
                imgname_2.append(data_list[j][1])

        
    df = pd.DataFrame(list(zip(similarities, imgname_1, imgname_2)),
                      columns=['Distance', 'ImageName 1', 'ImageName 2'])
    df.to_csv(csv_name, index=False)
    
    
def compare_two_different(real_list, synthetic_list, csv_name, all_=False):
    similarities = []
    imgname_1 = []
    imgname_2 = []
    
    if not all_: # If we are comparing positive against positive, and negative against negative
        # Negative
        neg_real_feat = get_fea([item[0] for item in real_list[0]])
        neg_fake_feat = get_fea([item[0] for item in synthetic_list[0]])
        for i in range(len(neg_real_feat)):
            for j in range(len(neg_fake_feat)):
                d = cosine_similarity(neg_real_feat[i], neg_fake_feat[j])
                similarities.append(d)
                imgname_1.append(real_list[0][i][1])
                imgname_2.append(synthetic_list[0][j][1])
        
        # Positive
        pos_real_feat = get_fea([item[0] for item in real_list[1]])
        pos_fake_feat = get_fea([item[0] for item in synthetic_list[1]])
        for i in range(len(pos_real_feat)):
            for j in range(len(pos_fake_feat)):
                d = cosine_similarity(pos_real_feat[i], pos_fake_feat[j])
                similarities.append(d)
                imgname_1.append(real_list[1][i][1])
                imgname_2.append(synthetic_list[1][j][1])
    
    elif all_: # If we are not making a distinction between positive and negative
        real_feat = get_fea([item[0] for item in real_list])
        fake_feat = get_fea([item[0] for item in synthetic_list])
        
        for i in range(len(real_feat)):
            for j in range(len(fake_feat)):
                d = cosine_similarity(real_feat[i], fake_feat[j])
                similarities.append(d)
                imgname_1.append(real_list[i][1])
                imgname_2.append(synthetic_list[j][1])

        
    df = pd.DataFrame(list(zip(similarities, imgname_1, imgname_2)),
                      columns=['Distance', 'ImageName 1', 'ImageName 2'])
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":

    print("Loading Images...\n")
    real_images_blood = load_real_images_by_class('/Users/rita/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Positive')
    real_images_no_blood = load_real_images_by_class('/Users/rita/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Negative')
    synthetic_images_blood = load_synthetic_images_by_class('/Users/rita/Qianyu_Code/Qianyu_Dataset/Synthetic_By_Batches/Positive')
    synthetic_images_no_blood = load_synthetic_images_by_class('/Users/rita/Qianyu_Code/Qianyu_Dataset/Synthetic_By_Batches/Negative')

    print("Split 3 batches...\n")
    # positive batches
    batch1_blood = filter_batch(synthetic_images_blood, batch="Batch1") # 23
    batch2_blood = filter_batch(synthetic_images_blood, batch="Batch2") # 146
    batch3_blood = filter_batch(synthetic_images_blood, batch="Batch3") # 298
    batch23_blood = batch2_blood+batch3_blood # 444
    all_synthetic_blood = batch1_blood+batch2_blood+batch3_blood # 467

    # negative batches
    batch1_no_blood = filter_batch(synthetic_images_no_blood, batch="Batch1") # 24
    batch2_no_blood = filter_batch(synthetic_images_no_blood, batch="Batch2") # 146
    batch3_no_blood = filter_batch(synthetic_images_no_blood, batch="Batch3") # 298
    batch23_no_blood = batch2_no_blood+batch3_no_blood # 444
    all_synthetic_no_blood = batch1_no_blood+batch2_no_blood+batch3_no_blood # 468
    
    print("Loading Model...\n")
    model = create_model()
    
    print("Random sample...\n")
    random.seed(427)
    # Use in R vs.R & S vs.S
    sample_real_images_no_blood = random.sample(real_images_no_blood, 113)
    sample_real_images_blood = random.sample(real_images_blood, 113)
    sample_all_synthetic_no_blood = random.sample(all_synthetic_no_blood, 113)
    sample_all_synthetic_blood = random.sample(all_synthetic_blood, 113)
    
    # Use in R vs.S
    small_real_images_no_blood = random.sample(real_images_no_blood, 80)
    small_real_images_blood = random.sample(real_images_blood, 80)
    small_synthetic_no_blood = random.sample(all_synthetic_no_blood, 80)
    small_synthetic_blood = random.sample(all_synthetic_blood, 80)
    
    # Use in R vs.Batches
    real_blood = random.sample(real_images_blood, len(batch1_blood))
    real_no_blood = random.sample(real_images_no_blood, len(batch1_no_blood))
    real_L2_blood = random.sample(real_images_blood, len(batch2_blood))
    real_L2_no_blood = random.sample(real_images_no_blood, len(batch2_no_blood))
    real_L3_blood = random.sample(real_images_blood, len(batch3_blood))
    real_L3_no_blood = random.sample(real_images_no_blood, len(batch3_no_blood))
    
    sample_batch1_blood = random.sample(batch1_blood, len(batch1_blood))
    sample_batch1_no_blood = random.sample(batch1_no_blood, len(batch1_no_blood))
    sample_batch2_blood = random.sample(batch2_blood, len(batch1_blood))
    sample_batch2_no_blood = random.sample(batch2_no_blood, len(batch1_no_blood))
    sample_batch3_blood = random.sample(batch3_blood, len(batch1_blood))
    sample_batch3_no_blood = random.sample(batch3_no_blood, len(batch1_no_blood))
    sample_batch23_blood = random.sample(batch23_blood, len(batch1_blood))
    sample_batch23_no_blood = random.sample(batch23_no_blood, len(batch1_no_blood))
    sample_small_all_no_blood = random.sample(all_synthetic_no_blood, len(batch1_no_blood))
    sample_small_all_blood = random.sample(all_synthetic_blood, len(batch1_blood))
    
    # Real vs. Synthetic
    print("distance RS...\n")
    # Real vs. Real
    compare_same_list([sample_real_images_no_blood, sample_real_images_blood], 'Real(blood)_Real(no_blood).csv') # 113
    # Syn. vs. Syn.
    compare_same_list([sample_all_synthetic_no_blood, sample_all_synthetic_blood], 'Synthetic(blood)_Synthetic(no_blood).csv') # 113
    # Real vs. Syn.
    compare_two_different([small_real_images_no_blood, small_real_images_blood], [small_synthetic_no_blood, small_synthetic_blood], 'Real_Synthetic.csv') # 80
    
    compare_two_different([real_no_blood, real_blood], [sample_small_all_no_blood, sample_small_all_blood], 'Real_Synthetic_Batches.csv')
    # Real vs. Syn. Batch 1
    compare_two_different([real_no_blood,real_blood], [sample_batch1_no_blood,sample_batch1_blood], 'Real_Synthetic_Batch1.csv')
    # Real vs. Syn. Batch 2
    compare_two_different([real_no_blood,real_blood], [sample_batch2_no_blood,sample_batch2_blood], 'Real_Synthetic_Batch2.csv')
    # Real vs. Syn. Batch 3
    compare_two_different([real_no_blood,real_blood], [sample_batch3_no_blood,sample_batch3_blood], 'Real_Synthetic_Batch3.csv')
    # Real vs. Syn. Batch 2&3
    compare_two_different([real_no_blood,real_blood], [sample_batch23_no_blood,sample_batch23_blood], 'Real_Synthetic_Batch23.csv')
    
    # Real vs. Syn. Batches All
    compare_two_different(real_no_blood+real_blood, sample_batch1_no_blood+sample_batch1_blood, 'Real_Synthetic_Batch1_All.csv', all_=True)
    compare_two_different(real_no_blood+real_blood, sample_batch2_no_blood+sample_batch2_blood, 'Real_Synthetic_Batch2_All.csv', all_=True)
    compare_two_different(real_no_blood+real_blood, sample_batch3_no_blood+sample_batch3_blood, 'Real_Synthetic_Batch3_All.csv', all_=True)
    compare_two_different(real_no_blood+real_blood, sample_batch23_no_blood+sample_batch23_blood, 'Real_Synthetic_Batch23_All.csv', all_=True)
    
    # Real vs. Augmentation
    print("distance RRaug...\n")
    real_aug_rotate_blood = []
    folder = '/Users/rita/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Positive'
    for filename in glob(folder+'/*'):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.png']]):
            img = image.load_img(filename, target_size=(224, 224))
            img = img.rotate(90)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            if img is not None:
                real_aug_rotate_blood.append([img, filename])
                
    real_aug_rotate_no_blood = []
    folder = '/Users/rita/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Negative'
    for filename in glob(folder+'/*'):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.png']]):
            img = image.load_img(filename, target_size=(224, 224))
            img = img.rotate(90)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            if img is not None:
                real_aug_rotate_no_blood.append([img, filename])
                
    compare_two_different([real_images_no_blood, real_images_blood], [real_aug_rotate_no_blood, real_aug_rotate_blood], 'Real_Rotate90.csv')
    
    real_aug_blood = []
    folder = '/Users/rita/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Positive'
    for filename in glob(folder+'/*'):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.png']]):
            img = image.load_img(filename, target_size=(224, 224))
            img = np.fliplr(img)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            if img is not None:
                   real_aug_blood.append([img, filename])
                    
    real_aug_no_blood = []
    folder = '/Users/rita/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Negative'
    for filename in glob(folder+'/*'):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.png']]):
            img = image.load_img(filename, target_size=(224, 224))
            img = np.fliplr(img)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            if img is not None:
                   real_aug_no_blood.append([img, filename])
                    
    compare_two_different([real_images_no_blood, real_images_blood], [real_aug_no_blood, real_aug_blood], 'Real_Flip.csv')
    
    real_aug_contr_blood = []
    folder = '/Users/rita/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Positive'
    for filename in glob(folder+'/*'):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.png']]):
            img = image.load_img(filename, target_size=(224, 224))
            enhancer = ImageEnhance.Contrast(img)
            factor = 2  # increase contrast
            img = enhancer.enhance(factor)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            if img is not None:
                real_aug_contr_blood.append([img, filename])
    
    real_aug_contr_no_blood = []
    folder = '/Users/rita/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Negative'
    for filename in glob(folder+'/*'):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg', '.png']]):
            img = image.load_img(filename, target_size=(224, 224))
            enhancer = ImageEnhance.Contrast(img)
            factor = 2  # increase contrast
            img = enhancer.enhance(factor)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            if img is not None:
                real_aug_contr_no_blood.append([img, filename])
    
    compare_two_different([real_images_no_blood, real_images_blood], [real_aug_contr_no_blood, real_aug_contr_blood], 'Real_Increase_Contrast.csv')
    
    print("All done!\n")
    