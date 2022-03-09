import pickle as pickle
from random import *
import numpy as np
from torchvision import transforms
import os
from collections import defaultdict
import sys, re
import pandas as pd
from PIL import Image
from types import *
import jieba
import os.path

import sys
if sys.version_info[0] >= 3:
    unicode = str

def stopwordslist(filepath = '/home/madm/Documents/EANN_recon/data/weibo/stop_words.txt'):
    stopwords = {}
    for line in open(filepath, 'r', encoding="UTF-8").readlines():
        line = line.encode("utf-8").strip()
        stopwords[line] = 1
    return stopwords

def clean_str_sst(string):
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


def read_image():
    image_list = {}
    file_list = ['/home/madm/Documents/EANN_recon/data/weibo/nonrumor_images/', '/home/madm/Documents/EANN_recon/data/weibo/rumor_images/'] 
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print(filename)
    print("image length " + str(len(image_list)))
    return image_list


def write_txt(data):
    f = open("/home/madm/Documents/EANN_recon/data/weibo/top_n_data.txt", 'wb')
    for line in data:
        for l in line:
            f.write(l+"\n")
        f.write("\n")
        f.write("\n")
    f.close()


text_dict = {}
def write_data(flag, image, text_only):

    def read_post(flag):
        stop_words = stopwordslist()
        pre_path = "/home/madm/Documents/EANN_recon/data/weibo/tweets/"
        file_list = [pre_path + "test_nonrumor.txt", pre_path + "test_rumor.txt", \
                         pre_path + "train_nonrumor.txt", pre_path + "train_rumor.txt"]
        if flag == "train":
            id = pickle.load(open("/home/madm/Documents/EANN_recon/data/weibo/train_id.pickle", 'rb'))
        elif flag == "validate":
            id = pickle.load(open("/home/madm/Documents/EANN_recon/data/weibo/validate_id.pickle", 'rb'))
        elif flag == "test":
            id = pickle.load(open("/home/madm/Documents/EANN_recon/data/weibo/test_id.pickle", 'rb'))


        post_content = []
        data = []
        column = ['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label']
        map_id = {}
        top_data = []
        for k, f in enumerate(file_list):

            f = open(f, 'rb')
            if (k + 1) % 2 == 1:
                label = 0 
            else:
                label = 1 

            twitter_id = 0
            line_data = []

            for i, l in enumerate(f.readlines()):
                if (i + 1) % 3 == 1:
                    line_data = []
                    twitter_id = l.split('|'.encode('UTF-8'))[0]
                    line_data.append(twitter_id)

                if (i + 1) % 3 == 2:

                    line_data.append(l.lower())

                if (i + 1) % 3 == 0:
                    l = clean_str_sst(str(l, encoding="utf-8"))

                    seg_list = jieba.cut_for_search(l)
                    new_seg_list = []
                    for word in seg_list:
                        if word not in stop_words:
                            new_seg_list.append(word)

                    clean_l = " ".join(new_seg_list)
                    if len(clean_l) > 10 and line_data[0].decode("UTF-8") in id:
                        post_content.append(l)
                        line_data.append(l)
                        line_data.append(clean_l)
                        line_data.append(label)
                        event = int(id[line_data[0].decode("UTF-8")])
                        if event not in map_id:
                            map_id[event] = len(map_id)
                            event = map_id[event]
                        else:
                            event = map_id[event]

                        line_data.append(event)

                        data.append(line_data)


            f.close()
        
        data_df = pd.DataFrame(np.array(data), columns=column)
        write_txt(top_data)

        return post_content, data_df

    post_content, post = read_post(flag)
    print("Original post length is " + str(len(post_content)))
    print("Original data frame is " + str(post.shape))

    def paired(text_only = False):
        ordered_image = []
        ordered_text = []
        ordered_post = []
        ordered_event= []
        label = []
        post_id = []
        image_id_list = []
        #image = []

        image_id = ""
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break

            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                ordered_text.append(post.iloc[i]['original_post'])
                ordered_post.append(post.iloc[i]['post_text'])
                ordered_event.append(post.iloc[i]['event_label'])
                post_id.append(id)


                label.append(post.iloc[i]['label'])

        label = np.array(label, dtype=np.int)
        ordered_event = np.array(ordered_event, dtype=np.int)

        print("Label number is " + str(len(label)))
        print("Rummor number is " + str(sum(label)))
        print("Non rummor is " + str(len(label) - sum(label)))

        data = {"post_text": np.array(ordered_post),
                "original_post": np.array(ordered_text),
                "image": ordered_image, "social_feature": [],
                "label": np.array(label), \
                "event_label": ordered_event, "post_id":np.array(post_id),
                "image_id":image_id_list}

        print("data size is " + str(len(data["post_text"])))
        
        return data

    paired_data = paired(text_only)

    print("paired post length is "+str(len(paired_data["post_text"])))
    print("paried data has " + str(len(paired_data)) + " dimension")
    return paired_data


def load_data(train, validate, test):
    vocab = defaultdict(float)
    all_text = list(train['post_text']) + list(validate['post_text'])+list(test['post_text'])
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text


def add_unknown_words(word_vecs, vocab, min_df=1, k=32):

    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_W(word_vecs, k=32):
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def get_data(text_only):

    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
        image_list = read_image()

    train_data = write_data("train", image_list, text_only)
    valiate_data = write_data("validate", image_list, text_only)
    test_data = write_data("test", image_list, text_only)

    print("loading data...")

    vocab, all_text = load_data(train_data, valiate_data, test_data)

    print("number of sentences: " + str(len(all_text)))
    print("vocab size: " + str(len(vocab)))
    max_l = len(max(all_text, key=len))
    print("max sentence length: " + str(max_l))


    word_embedding_path = "/home/madm/Documents/EANN_recon/data/weibo/w2v.pickle"

    w2v = pickle.load(open(word_embedding_path, 'rb+'), encoding='ISO-8859-1')

    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))

    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)

    W2 = {}
    w_file = open("/home/madm/Documents/EANN_recon/data/weibo/word_embedding.pickle", "wb")
    pickle.dump([W, W2, word_idx_map, vocab, max_l], w_file)
    w_file.close()
    return train_data, valiate_data, test_data