import torch
from torch.utils.data import Dataset
import numpy as np
from torch.autograd import Variable
import pickle as pickle
import copy

from process_weibo import *

class Transform_Numpy_Tensor(Dataset):
    def __init__(self, flag, dataset):
        self.text  = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        self.mask  = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print(flag,
              '数量统计',
              'text: %d, image: %d, label: %d, event_label: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label))
        )
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]

def Transform_Tensor_Variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def Transform_Tensor_Numpy(x):
    return x.data.cpu().numpy()

def load_data(args):
    # 分割数据集和embed
    train, validate, test = split_embed(args.text_only)

    word_vector_path = '/home/madm/Documents/EANN_recon/data/weibo/word_embedding.pickle'
    f = open(word_vector_path, 'rb')
    weight = pickle.load(f) 
    W, W2, word_idx_map, vocab, max_len = weight[0], weight[1], weight[2], weight[3], weight[4]
    args.vocab_size = len(vocab)
    args.max_len = max_len

    print("embedding验证集")
    word_embedding, mask = word2vec(validate['post_text'], word_idx_map, args)
    validate['post_text'] = word_embedding
    validate['mask'] = mask

    print("embedding测试集")
    word_embedding, mask = word2vec(test['post_text'], word_idx_map, args)
    test['post_text'] = word_embedding
    test['mask'] = mask

    print("embedding训练集")
    word_embedding, mask = word2vec(train['post_text'], word_idx_map, args)
    train['post_text'] = word_embedding
    train['mask'] = mask

    print("-"*50,"数据载入结束","-"*50)
    return train, validate, test, W