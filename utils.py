import re
import numpy as np
from collections import defaultdict
import copy

def stopwordslist():
    filepath = '/home/madm/Documents/EANN_recon/data/weibo/stop_words.txt'
    stopwords = {}
    for line in open(filepath, 'r', encoding="UTF-8").readlines():
        line = line.encode("utf-8").strip()
        stopwords[line] = 1
    return stopwords

def clean_str_sst(string):
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()

def sum_post(train, validate, test):
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

    for idx, word in enumerate(word_vecs):
        W[idx] = word_vecs[word]
        word_idx_map[word] = idx
    # W 只有词权重；word_idx_map 只有词
    return W, word_idx_map

def word2vec(post, word_id_map, args):
    word_embedding = []
    mask = []

    for sentence in post:
        sen_embedding = []
        mask_seq = np.zeros(args.max_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for word in sentence:
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.max_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))

    # word_embedding：根据word_id_map的词索引序列，其余补零；mask：有词位置填1，其余补零
    return word_embedding, mask