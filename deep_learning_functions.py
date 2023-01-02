import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime
import copy
import pandas as pd
import time
import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from constants import LABBELS_NAMES
from util_functions import *



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def save_summary( grid_elements, start_time, end_time, save_path):
    total_time = end_time - start_time
    #file = open(save_path+"/summary.txt","w+")
    #file.write(f"Total time: {total_time:.3f}\n")
    #file.close()


def random_init(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        m.weight.data=torch.randn(m.weight.size())*.01


def train(net, trainloader, optimizer, criterion , device):
    net.train()
    running_loss = 0.0
    running_acc = 0.0
    count_minibatchs = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        running_acc += (preds == labels).sum().item()

        count_minibatchs = i+1
        
    epoch_loss = running_loss / count_minibatchs
    epoch_acc = 100. * (running_acc / len(trainloader.dataset))
    print(f"loss: {epoch_loss:.3f}, accuracy: {epoch_acc:.3f}")
    return epoch_loss, epoch_acc


def validate(net, validloader, criterion, device):
    net.eval()
    running_loss = 0.0
    running_acc = 0.0
    count_minibatchs = 0
    with torch.no_grad():
        for i, data in enumerate(validloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            running_acc += (preds == labels).sum().item()

            count_minibatchs = i+1
    epoch_loss = running_loss / count_minibatchs
    epoch_acc = 100. * (running_acc / len(validloader.dataset))
    print(f"val_loss: {epoch_loss:.3f}, val_accuracy: {epoch_acc:.3f}")
    return epoch_loss, epoch_acc


def test(net, testloader, device):
    net.eval()
    y_pred, y_conf, y_original, y_names = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels

            # get original labbels
            y_original.extend(labels.numpy())

            # forward
            outputs=net(inputs)

            conf, preds = torch.max(outputs.data, 1)
            preds = preds.cpu()
            conf = conf.cpu()
            y_pred.extend(preds.numpy())
            y_conf.extend(conf.numpy())
            name = testloader.dataset.samples[i][0]
            y_names.append( name )

    return y_pred, y_conf,y_original,y_names


def load_model( net, model_state, device):
    net.load_state_dict( model_state )
    net.to(device)
    return net

    
def save_acc_plot(train_acc, valid_acc, model_save_path):
    try:
        #plt.figure()
        #plt.plot(train_acc, color='red', label='train acc')
        #plt.plot(valid_acc, color='blue', label='valid acc')
        #plt.xlabel('Epochs')
        #plt.ylabel('Accuracy')
        #plt.legend()
        #plt.savefig(model_save_path+'/accuracy.png')
        return True
    except:
        return False

def save_loss_plot(train_loss, valid_loss, model_save_path):
    try:
        #plt.figure()
        #plt.plot(train_loss, color='red', label='train loss')
        #plt.plot(valid_loss, color='blue', label='valid loss')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.savefig(model_save_path+'/loss.png')    
        return True
    except:
        return False


def get_image_name( i_name ):
    arr = i_name.split('/')
    for element in arr:
        if element.startswith('ch00'):
            return element
    return ''

def save_confusion_matrix(y_true, y_conf, y_pred, y_names, model_save_path):
    cf_matrix = confusion_matrix(y_true, y_pred, normalize=None)
    accuracy = accuracy_score(y_true, y_pred )
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    #file= open(model_save_path+"/general_scores.txt","w+")
    #file.write(f"Accuracy Score: {accuracy:.3f}\n")
    #file.write(f"Precision Score: {precision:.3f}\n")
    #file.write(f"Recall Score: {recall:.3f}\n")
    #file.write(f"F1 Score: {f1:.3f}\n")
    #file.close()
    #file= open(model_save_path+"/confusion_matrix.txt","w+")
    #for elements in cf_matrix:
    #    for element in elements:
    #        file.write(str(element)+' ')
    #    file.write( '\n' )
    #file.close()

    #file= open(model_save_path+"/predicted_values.txt","w+")
    #for i in range(len(y_true)):
    #    image_name = get_image_name(y_names[i])
    #    file.write(image_name+' ')
    #    file.write(str(y_true[i])+' ')
    #    file.write(str(y_pred[i])+' ')
    #    file.write(str(y_conf[i])+' ')
    #    file.write( '\n' )
    #file.close()

    #cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    #df_cm = pd.DataFrame(cf_matrix, index = [class_name for class_name in LABBELS_NAMES], columns = [class_name for class_name in LABBELS_NAMES])
    #plt.figure()
    #sn.heatmap(df_cm, annot=True, cmap='Blues')
    #plt.tight_layout()
    #plt.savefig(model_save_path+'/confusion_matrix.png')
    return accuracy, precision, recall, f1, cf_matrix


def process_net( net, epochs, trainloader, validloader, testloader, optimizer, criterion, model_save_path):
    aux = copy.deepcopy(net)
    train_loss, valid_loss ,train_acc, valid_acc = [],[],[],[]
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = net.to(device)
    best_vloss = float('inf')
    best_model_state= []

    early_stopper = EarlyStopper( patience = 20 )

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_epoch_loss, train_epoch_acc = train(net, trainloader, optimizer, criterion,device)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        valid_epoch_loss, valid_epoch_acc = validate(net, validloader, criterion,device)        
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)

        if valid_epoch_loss < best_vloss:
            best_vloss = valid_epoch_loss
            best_model_state = net.state_dict()
        if early_stopper.early_stop(valid_epoch_loss):
            print( 'Stop due Early stopping\n' )
            break
        print("\n\n")
        
    best_net = load_model( aux, best_model_state, device)
    y_pred, y_conf, y_true, y_names= test(best_net, testloader, device)
    save_acc_plot(train_acc, valid_acc, model_save_path)
    save_loss_plot(train_loss, valid_loss, model_save_path)
    accuracy, precision, recall, f1, _ = save_confusion_matrix(y_true, y_conf, y_pred, y_names, model_save_path)
    plt.close('all')
    return accuracy, precision, recall, f1


def load_data_pytorch_format( dataset_path, batch_size, image_dimention ):
    
    transform = transforms.Compose([
        transforms.Resize(image_dimention[0]),
        transforms.ToTensor()])
    
    train_dataset = datasets.ImageFolder(root=dataset_path+'/train', 
                                         transform=transform)
    valid_dataset = datasets.ImageFolder(root=dataset_path+'/valid',
                                         transform=transform)
    test_dataset = datasets.ImageFolder(root=dataset_path+'/test',
                                        transform=transform)    

    train_loader = DataLoader( train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=4, 
                              pin_memory=True)
    valid_loader = DataLoader( valid_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=4, 
                              pin_memory=True)
    test_loader = DataLoader( test_dataset, 
                             batch_size=1, 
                             shuffle=False,
                             num_workers=4, 
                             pin_memory=True)

    return train_loader, valid_loader, test_loader
