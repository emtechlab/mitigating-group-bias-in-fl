import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pylab
import torch.nn.functional as F
import os
import csv
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import math
import copy

from copy import deepcopy
from torch import nn
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(args, client_id, cm, classes, epoch ,normalize=True, title='', cmap=plt.cm.Blues):
   
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """

   FONTSIZE = 25
   
   cm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

   #row_sums = np.nansum(cm, axis=1, keepdims=True)
   #cm = np.divide(cm, row_sums, where=row_sums!=0)

   base_name = f"/home/{os.getenv('USER')}/Mitigating-Group-Bias-in-FL/evaluation/{args.federated_type}/{args.dataset}/train_matrix"

   # Generate the file name with the number appended
   file_name = f'{base_name}{client_id}{epoch}.npy'
 
   # Save the NumPy array to a file
   np.save(file_name, cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.

   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=FONTSIZE-8)

   plt.tight_layout()
   plt.ylabel('True label', fontsize=FONTSIZE-6)
   plt.xlabel('Predicted label', fontsize=FONTSIZE-6)

   plt.xticks(fontsize=FONTSIZE-6)
   plt.yticks(fontsize=FONTSIZE-6)

   cm_dest = base_name + str(client_id) + str(epoch) + ".png"

   pylab.savefig(cm_dest, bbox_inches='tight', pad_inches=0)

def update_csv(args, client_idx, test_loss, test_acc, avg_grad_norm):

    filename = f"/home/{os.getenv('USER')}/Mitigating-Group-Bias-in-FL/evaluation/{args.federated_type}/{args.dataset}/client_train{client_idx}.csv"

    # Check if the file exists, create it if it doesn't
    if not os.path.exists(filename):
        
        with open(filename, 'w', newline='') as file:
            
            writer = csv.DictWriter(file, fieldnames=['Loss', 'Accuracy', 'Norm'])
            writer.writeheader()

    # Append the new data to the file
    with open(filename, 'a', newline='') as file:
        
        writer = csv.DictWriter(file, fieldnames=['Loss', 'Accuracy', 'Norm'])

        # Write a new row with test_loss and test_acc
        writer.writerow({'Loss': test_loss, 'Accuracy': test_acc, 'Norm': avg_grad_norm})

# Training
def update_weights(args, model, device, global_epoch, train_loader, testDataloader, criterion, client_id):

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    frz_model = deepcopy(model)

    model.train()

    if (args.dataset in ['acsemployment', 'acsincome', 'acspubliccoverage', 'acsmobility']):

        actual_labels = torch.tensor([], dtype=torch.float32, device=device)
        test_preds = torch.tensor([], dtype=torch.float32, device=device)
    else:
        actual_labels = torch.tensor([])
        test_preds = torch.tensor([])

    for epoch in range(1, args.local_epochs+1):

        train_loss = 0
        correct = 0
        total = 0
        total_grad_norm = 0.0
    
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.to(device), targets.to(device)

            actual_labels = torch.cat((actual_labels.to(device), targets.type(torch.float).to(device)), dim=0)
             
            optimizer.zero_grad()
             
            outputs = model(inputs)
             
            loss = criterion(outputs, targets)

            test_preds = torch.cat((test_preds.to(device), outputs) ,dim=0)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if (args.dataset in ['acsemployment', 'acsincome', 'acspubliccoverage', 'acsmobility']):

                predicted = (outputs >= 0.5).float()
            else:
                _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Compute average gradient norm across all batches
        avg_grad_norm = total_grad_norm / max(batch_idx+1, 1)

        #preds_correct = test_preds.argmax(dim=1).eq(actual_labels).sum().item()

        if (args.dataset in ['acsemployment', 'acsincome', 'acspubliccoverage', 'acsmobility']):
            cm = confusion_matrix(actual_labels.cpu().numpy(), (test_preds >= 0.5).cpu().numpy())
        else:
            cm = confusion_matrix(actual_labels.detach().cpu().numpy(), test_preds.argmax(dim=1).detach().cpu().numpy())

        if (args.dataset == 'mnist'):
            classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        if (args.dataset == 'fmnist'):
            classes = ("Boot", "Bag", "Coat", "Dress", "Pullover", "Sandal", "Shirt", "Sneaker", "Trouser", "T-shirt")
        if (args.dataset == 'utk'):
            classes = ('female', 'male')
        if (args.dataset == 'fer'):
            classes = ('Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
        if (args.dataset in ['acsemployment', 'acsincome', 'acspubliccoverage', 'acsmobility']):
            classes = ('0', '1')
        if (args.dataset == 'cifar10'):
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        update_csv(args, client_id, train_loss/(batch_idx+1), 100.*correct/total, avg_grad_norm)

        plt.figure(figsize=(10,9))
        plot_confusion_matrix(args, client_id, cm, classes, global_epoch)
        
        print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f}|'.format(
                global_epoch,args.global_epochs, client_id, epoch, args.local_epochs,train_loss/(batch_idx+1), 100.*correct/total))
         
    return copy.deepcopy(model.state_dict()), model, avg_grad_norm
