import numpy as np
import pickle
import hickle
import time
import os
import sys
import json

from scipy import ndimage
# from collections import Counter
# from core.vggnet import Vgg19
# from core.utils import *
from PIL import Image
import tensorflow as tf

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import pickle

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)


data_path = './data/train'
split = "train"
data = {}
np.set_printoptions(threshold = sys.maxsize)
pd.set_option('display.max_rows', None)

features = hickle.load('/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/our_data/test.features.hkl')
print(features.shape)
# # data['features'] = data['features'][:1000]
# #print(data['features'][:5])
# print(data['features'].shape)
#
# with open(os.path.join(data_path, '%s.file.names.pkl' % split), 'rb') as f:
#     data['file_names'] = pickle.load(f)
#     # data['file_names'] = data['file_names'][:1000]
#     #print(data['file_names'][:5])
#     print(data['file_names'].shape)
#
# with open(os.path.join(data_path, '%s.captions.pkl' % split), 'rb') as f:
#     data['captions'] = pickle.load(f)
#     # data['captions'] = data['captions'][:1000]
#     #print(data['captions'][:5])
#     print(data['captions'].shape)

# with open(os.path.join(data_path, '%s.image.idxs.pkl' % split), 'rb') as f:
#     data['image_idxs'] = pickle.load(f)
#     # data['image_idxs'] = data['image_idxs'][:1000]
#     print(data['image_idxs'])
#     print(data['image_idxs'][:5].shape)

# with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
#     data['word_to_idx'] = pickle.load(f)
#     print(len(data['word_to_idx']))
# caption_file = 'data/annotations/captions_train2014.json'
# with open(caption_file) as f:
#     caption_data = json.load(f)
# id_to_filename = {image['id']: image['file_name'] for image in caption_data['images']}
# print(id_to_filename)

def get_featrues():
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        image_batch_file = []

        for image in os.listdir(r".\image_data_to_be_labeled\resized_image"):

            image_batch_file.append(os.path.join(r".\image_data_to_be_labeled\resized_image",image.rstrip('\n')))
        print(len(image_batch_file))
        # for image in image_batch_file:
        #     img = Image.open(image)
        #     img = img.resize((224,224))
        #     img.save(os.path.join(r".\image_data_to_be_labeled\resized_image",os.path.basename(image)))
        f = open(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\train_list.txt','w')
        f.close()
        f = open(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\test_list.txt','w')
        f.close()
        train_batch_file = image_batch_file[:220]
        with open(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\train_list.txt','a') as f:
            for train in train_batch_file:
                f.write(train+'\n')

        test_batch_file = image_batch_file[220:]
        with open(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\test_list.txt','a') as f:
            for test in test_batch_file:
                f.write(test+'\n')

        train_feats = np.ndarray([len(train_batch_file), 196, 512], dtype=np.float32)
        test_feats = np.ndarray([len(test_batch_file), 196, 512], dtype=np.float32)
        train_batch = []
        test_batch = []
        for image in train_batch_file:
            image_read = ndimage.imread(image, mode='RGB').astype(np.float32)
            train_batch.append(image_read)

        for image in test_batch_file:
            image_read = ndimage.imread(image, mode='RGB').astype(np.float32)
            test_batch.append(image_read)

        train_batch = np.array(train_batch)
        test_batch = np.array(test_batch)

        print(train_batch.shape)
        print(test_batch.shape)
        train_feats = np.ndarray([220, 196, 512], dtype=np.float32)
        test_feats = np.ndarray([70, 196, 512], dtype=np.float32)
        for i in range(22):
            train_feats[i*10:(i+1)*10] = sess.run(vggnet.features, feed_dict={vggnet.images: train_batch[i*10:(i+1)*10]})

        for j in range(7):
            test_feats[j*10:(j+1)*10] = sess.run(vggnet.features, feed_dict={vggnet.images: test_batch[j*10:(j+1)*10]})

        print(train_feats.shape)
        print(test_feats.shape)

    # use hickle to save huge feature vectors
    hickle.dump(train_feats, r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\\our_data\train.features.hkl")
    hickle.dump(test_feats, r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\\our_data\test.features.hkl")


def generate_lists():
    file = open(r'.\image_data_to_be_labeled\test_list.txt')
    list = file.readlines()
    test_list = []
    i = 0
    for line in list:
        test_list.append(os.path.basename(line).rstrip('\n'))

    print(len(test_list))
    test_list = np.array(test_list)
    save_pickle(test_list,
                r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\\our_data\test.file.names.pkl")


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples, max_length + 2)).astype(np.int32)

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ")  # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
    print("Finished building caption vectors")
    return captions


def generate_captions():
    df = pd.read_csv('/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/Label.csv')
    print("Column headings:")
    print(df.head(10))

    with open('/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/our_data/test.file.names.pkl','rb') as file:

        list = pickle.load(file)
    print(len(list))
    with open('/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)

    word_to_idx[','] = len(word_to_idx)
    print(word_to_idx)
    max_length=25
    captions = np.ndarray((len(list), max_length + 2)).astype(np.int32)
    i = 0
    for name in list:
        train = os.path.basename(name).rstrip('\n')
        print(train)
        ind = df[df['Image_Name']==train].index[0]
        caption = df['Label'][ind]

        caption = caption.replace('.', '').replace("'", "").replace('"', '').replace(",", " ,")
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').lower()
        caption = " ".join(caption.split())# replace multiple spaces
        print(caption)
        words = caption.split(" ")  # caption contains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])

        if len(cap_vec) > (max_length + 2):
            print("exceed length")
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions[i, :] = np.asarray(cap_vec)
        i += 1
    print(captions.shape)
    save_pickle(captions,'/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/our_data/test.captions.pkl')

def save_new_words():
    with open('/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)

    word_to_idx[','] = len(word_to_idx)
    print(len(word_to_idx))
    save_pickle(word_to_idx,'/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/our_data/word_to_idx.pkl')

# train = np.arange(220)
# save_pickle(train,'/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/our_data/train.image.idxs.pkl')
# print(train.shape)
#
# test = np.arange(70)
# save_pickle(test,'/Users/stephaniexia/Documents/UM/S2/CIS5700 Advanced data mining/Project/our_data/test.image.idxs.pkl')

