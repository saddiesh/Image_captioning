import numpy as np
import pickle
import hickle
import time
import os
import sys
import json

from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *
from PIL import Image
import tensorflow as tf

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import pickle

data_path = './data/train'
split = "train"
data = {}
np.set_printoptions(threshold = sys.maxsize)
pd.set_option('display.max_rows', None)
#
# data['features'] = hickle.load(os.path.join(data_path, '%s.features.hkl' % split))
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

working_dir = r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature'

def get_featrues():
    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    img_path = r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\JPEGImages"
    resized_img_path = r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\resized'
    for image in os.listdir(img_path):

        pil_im = Image.open(os.path.join(img_path,image))
        size = 224,224
        pil_im = pil_im.resize(size)
        pil_im.save(os.path.join(resized_img_path, image))


    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        image_batch_file = []

        for image in os.listdir(r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\resized"):

            image_batch_file.append(os.path.join(r".\image_data_to_be_labeled\Object_feature\resized",image.rstrip('\n')))
        print(len(image_batch_file))
        # for image in image_batch_file:
        #     img = Image.open(image)
        #     img = img.resize((224,224))
        #     img.save(os.path.join(r".\image_data_to_be_labeled\resized_image",os.path.basename(image)))
        f = open(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\train_list.txt','w')
        f.close()
        # f = open(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\test_list.txt','w')
        # f.close()
        train_batch_file = image_batch_file.copy()
        with open(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\train_list.txt','a') as f:
            for train in train_batch_file:
                f.write(train+'\n')
        #
        # test_batch_file = image_batch_file[220:]
        # with open(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\test_list.txt','a') as f:
        #     for test in test_batch_file:
        #         f.write(test+'\n')

        train_feats = np.ndarray([len(train_batch_file), 196, 512], dtype=np.float32)
        # test_feats = np.ndarray([len(test_batch_file), 196, 512], dtype=np.float32)
        train_batch = []
        # test_batch = []
        for image in train_batch_file:
            image_read = ndimage.imread(image, mode='RGB').astype(np.float32)
            train_batch.append(image_read)

        # for image in test_batch_file:
        #     image_read = ndimage.imread(image, mode='RGB').astype(np.float32)
        #     test_batch.append(image_read)

        train_batch = np.array(train_batch)
        # test_batch = np.array(test_batch)

        print(train_batch.shape)
        # print(test_batch.shape)
        # train_feats = np.ndarray([220, 196, 512], dtype=np.float32)
        # test_feats = np.ndarray([70, 196, 512], dtype=np.float32)
        for i in range(22):
            train_feats[i*10:(i+1)*10] = sess.run(vggnet.features, feed_dict={vggnet.images: train_batch[i*10:(i+1)*10]})

        # for j in range(7):
        #     test_feats[j*10:(j+1)*10] = sess.run(vggnet.features, feed_dict={vggnet.images: test_batch[j*10:(j+1)*10]})

        print(train_feats.shape)
        # print(test_feats.shape)

    # use hickle to save huge feature vectors
    hickle.dump(train_feats, r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\our_data\train.features.hkl")
    # hickle.dump(test_feats, r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\\our_data\test.features.hkl")



def generate_lists():
    with open(os.path.join(working_dir, 'train_list.txt')) as file:
        list = file.readlines()
    train_list = []
    for line in list:
        train_list.append(line.rstrip('\n'))

    print(len(train_list))
    train_list = np.array(train_list)
    save_pickle(train_list, os.path.join(working_dir,r'our_data\train.file.names.pkl'))

    # file = open(r'.\image_data_to_be_labeled\train_list.txt')
    # list = file.readlines()
    # train_list = []
    # for line in list:
    #     train_list.append(line.rstrip('\n'))
    #
    # print(len(train_list))
    # test_list = np.array(train_list)
    # save_pickle(train_list,
    #             r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\\our_data\train.file.names.pkl")


def generate_captions():
    df = pd.read_csv(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\Label_revised_2.csv')
    print("Column headings:")
    print(df.head(10))

    with open(r".\image_data_to_be_labeled\Object_feature\our_data\train.file.names.pkl",'rb') as file:

        list = pickle.load(file)
    print(len(list))
    with open(r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\our_data\train_own_voc\train\word_to_idx.pkl", 'rb') as f:
        word_to_idx = pickle.load(f)

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
    save_pickle(captions,r'.\image_data_to_be_labeled\Object_feature\our_data\train.captions.pkl')


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations):
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1

    print(word_to_idx[','])
    print(len(word_to_idx))
    print("Max length of caption: ", max_len)
    return word_to_idx



def generate_wordtoidx():
    df = pd.read_csv(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Label.csv')
    print("Column headings:")
    print(df.head(10))

    file = open(r'.\image_data_to_be_labeled\total_list.txt')
    list = file.readlines()
    captions = []
    for name in list:
        train = os.path.basename(name).rstrip('\n')
        ind = df[df['Image_Name']==train].index[0]
        caption = df['Label'][ind]

        caption = caption.replace('.', '').replace(',', ' ,').replace("'", "").replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ').lower()
        caption = " ".join(caption.split())# replace multiple spaces

        captions.append(caption)

    print(captions)

    word_to_idx = _build_vocab(captions, threshold=1)
    print(word_to_idx)
    save_pickle(word_to_idx,r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\our_data\train_own_voc\word_to_idx_rebuilt.pkl")



def generate_references():
    df = pd.read_csv(r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\Label_revised_2.csv')
    print("Column headings:")
    print(df.head(10))

    with open(r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\our_data\train\train.file.names.pkl",'rb') as file:

        list = pickle.load(file)
    print(len(list))

    max_length=25
    references = []
    i = 0
    for name in list:
        train = os.path.basename(name).rstrip('\n')
        print(train)
        ind = df[df['Image_Name']==train].index[0]
        ref = df['Label'][ind]
        ref = ref.lower()
        ref = " ".join(ref.split())# replace multiple spaces
        print(ref)
        references.append(ref)

    references = np.array(references)
    print(references.shape)
    save_pickle(references,r'C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\our_data\train\train.references.pkl')

def stack_feature():
    with open(os.path.join(working_dir, 'train_list.txt')) as file:
        list = file.readlines()

    objects = []
    for name in list:
        img_npy = os.path.join(r'.\image_data_to_be_labeled\Object_feature\Feature' , os.path.basename(name.rstrip('\n').rstrip('.jpg'))+'.npy')
        obj_features = np.load(img_npy)
        objects.append(obj_features[:,5:])
    objects = np.array(objects)
    objects = objects.reshape((-1,20,512))
    vgg_features = hickle.load(
        r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\our_data\train\train.vggfeatures.hkl")
    print(vgg_features.shape)
    print(objects.shape)
    whole_feature = np.concatenate((vgg_features,objects),axis=1)
    print(whole_feature.shape)
    hickle.dump(whole_feature,
                r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\image_data_to_be_labeled\Object_feature\our_data\train.features.hkl")

#generate_wordtoidx()

# generate_captions()
#generate_lists()
# features = hickle.load(r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\our_data\test\test.features.hkl")
# print(features)
# with open(r"C:\Users\song\Desktop\511project\show-attend-and-tell-tensorflow\data\train\train.references.pkl",'rb') as file:
#     list = pickle.load(file)
# print(list)
generate_references()
# get_featrues()
# generate_lists()

#stack_feature()