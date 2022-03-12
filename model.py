import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as functional

import sys
sys.path.append('/home/madm/Documents/EANN_recon/src/')

# 导入失败时重启下ssh
from gradreverse import *

class EANN(nn.Module):
    def __init__(self, args, W):
        super(EANN, self).__init__()

        self.args = args
        self.event_num = args.event_num
        self.hidden_size = args.hidden_dim

        """Embedding"""
        emb_dim = args.embed_dim 
        vocab_size = args.vocab_size
        # 创建一个词嵌入模型 权重是W
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))

        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        
        """Text_Part"""
        self.text_model = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        """Vision_Part"""
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vision_model = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)

        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        """Classifer_Part"""
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('classifer_fc1', nn.Linear(2 * self.hidden_size, 2))
        self.class_classifier.add_module('classifer_softmax', nn.Softmax(dim=1))

        """Discriminator_Part"""
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('discriminator_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('discriminator_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('discriminator_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('discriminator_softmax', nn.Softmax(dim=1))
    
    def forward(self, text, image, mask):

        """Image"""
        image = self.vision_model(image)
        image = functional.leaky_relu(self.image_fc1(image))

        """Text"""
        text_embeded = self.embed(text)
        text_masked = text_embeded * mask.unsqueeze(2).expand_as(text_embeded)
        text = text_masked.unsqueeze(1)
        text = [functional.leaky_relu(conv(text)).squeeze(3) for conv in self.text_model]
        text = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = functional.leaky_relu(self.fc1(text))

        """Concat"""
        text_image = torch.cat((text, image), 1)

        """Classifer"""
        class_output = self.class_classifier(text_image)

        """Discriminator"""
        reverse_feature = grad_reverse(text_image, self.args)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output