import time, os
import torch
import numpy as np
from sklearn import metrics

from data_loader import *


def train(args, optimizer, criterion, train_loader, model, validate_loader):

    best_validate_acc = 0.000
    best_validate_dir = ''
    
    for epoch in range(args.num_epochs):
        p = float(epoch) / 100
        
        lr = 0.001 / (1. + 10 * p) ** 0.75

        optimizer.lr = lr
        
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        vali_cost_vector = []

        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_image, train_mask, train_labels, event_labels = \
                Transform_Tensor_Variable(train_data[0]), Transform_Tensor_Variable(train_data[1]), Transform_Tensor_Variable(train_data[2]), \
                Transform_Tensor_Variable(train_labels), Transform_Tensor_Variable(event_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, domain_outputs = model(train_text, train_image, train_mask)

            ## Fake or Real loss
            class_loss = criterion(class_outputs, train_labels.long())

            # Event Loss
            domain_loss = criterion(domain_outputs, event_labels.long())
            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            accuracy = (train_labels == argmax.squeeze()).float().mean()
    
            class_cost_vector.append(class_loss.data.item())
            domain_cost_vector.append(domain_loss.data.item())
            cost_vector.append(loss.data.item())
            acc_vector.append(accuracy.data.item())

        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_image, validate_mask, validate_labels, event_labels = \
                Transform_Tensor_Variable(validate_data[0]), Transform_Tensor_Variable(validate_data[1]), Transform_Tensor_Variable(validate_data[2]), \
                Transform_Tensor_Variable(validate_labels), Transform_Tensor_Variable(event_labels)
            validate_outputs, domain_outputs = model(validate_text, validate_image, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            vali_loss = criterion(validate_outputs, validate_labels.long())
            
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.data.item())
            
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()
        print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
              % (
                  epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector),
                  np.mean(domain_cost_vector),
                  np.mean(acc_vector), validate_acc))

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)

            best_validate_dir = args.output_file + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), best_validate_dir)
        

def test(args, best_validate_dir, test_loader, model, W):

    print('testing model')
    model = model(args, W)
    model.load_state_dict(torch.load(best_validate_dir))

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_image, test_mask, test_labels = Transform_Tensor_Variable(
            test_data[0]), Transform_Tensor_Variable(test_data[1]), Transform_Tensor_Variable(test_data[2]), Transform_Tensor_Variable(test_labels)
        test_outputs, domain_outputs = model(test_text, test_image, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = Transform_Tensor_Nupy(test_outputs.squeeze())
            test_pred = Transform_Tensor_Nupy(test_argmax.squeeze())
            test_true = Transform_Tensor_Nupy(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, Transform_Tensor_Nupy(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, Transform_Tensor_Nupy(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, Transform_Tensor_Nupy(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))