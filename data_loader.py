import torch
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable
import pickle as pickle
import copy

import process_weibo as process_data

"""transform numpy to tensor"""
class Transform_Numpy_Tensor(Dataset):
    def __init__(self, dataset):
        self.text  = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        self.mask  = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print(
            '数量统计',
            'TEXT: %d, Image: %d, labe: %d, Event: %d'
            % (len(self.text), len(self.image), len(self.label), len(self.event_label))
        )
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]

"""transform tensor to variable"""
def Transform_Tensor_Variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

"""transform tensor to numpy"""
def Transform_Tensor_Nupy(x):
    return x.data.cpu().numpy()

"""load data"""
def load_data(args):
    train, validate, test = process_data.get_data(args.text_only)

    word_vector_path = '/home/madm/Documents/EANN_recon/data/weibo/word_embedding.pickle'
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f) 
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]
    args.vocab_size = len(vocab)
    args.sequence_len = max_len

    print("translate data to embedding")
    word_embedding, mask = word2vec(validate['post_text'], word_idx_map, args)
    validate['post_text'] = word_embedding
    validate['mask'] = mask

    print("translate test data to embedding")
    word_embedding, mask = word2vec(test['post_text'], word_idx_map, args)
    test['post_text'] = word_embedding
    test['mask'] = mask

    word_embedding, mask = word2vec(train['post_text'], word_idx_map, args)
    train['post_text'] = word_embedding
    train['mask'] = mask

    print("sequence length " + str(args.sequence_length))
    print("Train Data Size is " + str(len(train['post_text'])))
    print("Finished loading data ")
    return train, validate, test, W


def word2vec(post, word_id_map, args):
    word_embedding = []
    mask = []

    for sentence in post:
        sen_embedding = []
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for word in sentence:
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))

    return word_embedding, mask