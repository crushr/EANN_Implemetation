import torch
from torch.utils.data import DataLoader
import argparse

from data_loader import *
from model import *
from train_val import *
from parse_argument import *

def main(args):

    print('loading data')

    train_set, validation_set, test_set, W = load_data(args)

    train_dataset = Transform_Numpy_Tensor(train_set)
    validate_dataset = Transform_Numpy_Tensor(validation_set)
    test_dataset = Transform_Numpy_Tensor(test_set)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    print('building model')
    model = EANN(args, W)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                        list(model.parameters())),
                                lr=args.learning_rate)


    print("loader size " + str(len(train_loader)))

    print('start training')
    train(args, optimizer, criterion, train_loader, model, validate_loader)
    # test(args, best_validate_dir, test_loader, model, W)

    

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    # train = ''
    # test = ''
    output = '/home/madm/Documents/EANN_recon/RESULT/'
    args = parser.parse_args([output])

    main(args)