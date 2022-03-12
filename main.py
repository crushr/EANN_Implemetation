import torch
from torch.utils.data import DataLoader
import argparse
from torchsummary import summary

from data_loader import *
from model import *
from train_val import *
from parse_argument import *

def main(args):

    print("-"*50,"开始载入数据","-"*50)
    train_set, validation_set, test_set, W = load_data(args)

    train_dataset = Transform_Numpy_Tensor("trainset",train_set)
    validate_dataset = Transform_Numpy_Tensor("valset",validation_set)
    test_dataset = Transform_Numpy_Tensor("testset",test_set)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print("-"*50,"开始生成模型","-"*50)
    model = EANN(args, W)
    # torch.save(model, "./model_viz.pt")
    
    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    # 利用filter(lambda p: p.requires_grad, list(model.parameters())过滤不需要训练的参数部分
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())), 
                                 lr=args.learning_rate)

    print("bach总数：" , "训练集" + str(len(train_loader)) , "验证集" + str(len(validate_loader)) , "测试集" + str(len(test_loader)))

    print("-"*50,"开始训练","-"*50)

    '''使用tensorboard可视化，存在bug，tensorboard不支持网络中存在int型变量，只支持tensor'''
    # dataiter = iter(train_loader)
    # train_data, train_labels, event_labels = dataiter.next()
    # train_text, train_image, train_mask = Transform_Tensor_Variable(train_data[0]), Transform_Tensor_Variable(train_data[1]), Transform_Tensor_Variable(train_data[2])
    # writer = SummaryWriter('/home/madm/Documents/EANN_recon/experiment_1')
    # writer.add_graph(model,(train_text, train_image, train_mask))
    # writer.flush()
    '''使用torch.summary进行可视化，存在bug，要求模型传入int，但是传入了tensor'''
    # print(summary(model, [(363,), (3, 224, 224), (363,)], batch_size=20)) 
    '''综合上述两种方法，还是直接只用print(model)来的简单有效'''

    # train(args, optimizer, criterion, train_loader, model, validate_loader)
    test(args, test_loader, model, W)  

if __name__ == '__main__':
    '''
        argparse用法：
            1、创建ArgumentParser()对象
            2、调用add_argument()方法添加参数
            3、使用 parse_args()解析添加的参数
    '''
    parser = argparse.ArgumentParser()
    parser = parse_arguments(parser)
    args = parser.parse_args()
    main(args)