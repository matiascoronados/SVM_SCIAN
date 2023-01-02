from deep_learning_functions import *
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split as data_split
import torchvision.datasets as datasets
from sklearn.svm import SVC as SVM
from sklearn.svm import *
from sklearn.metrics import classification_report
from argparse import ArgumentParser
torch.multiprocessing.set_sharing_strategy('file_system')

torch.cuda.set_device(1)

def load_data( dataset_path, batch_size, image_dimention ):
    
    transform = transforms.Compose([
        transforms.Resize(image_dimention[0]),
        transforms.ToTensor()])
    
    dataset_train = datasets.ImageFolder(root=dataset_path+'/train', 
                                         transform=transform)    
    dataset_test = datasets.ImageFolder(root=dataset_path+'/test', 
                                         transform=transform)  

    dataset_train_loader = DataLoader( dataset_train, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=1, 
                              pin_memory=True)
    dataset_test_loader = DataLoader( dataset_test, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=1, 
                              pin_memory=True)
    return dataset_train_loader, dataset_test_loader


def get_features(net, loader, device):
    net.eval()
    all_features = []
    with torch.no_grad():
        for i, data in enumerate(loader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels
            outputs=net(inputs).data
            features = outputs.reshape((1, 512)).tolist()[0]
            name = loader.dataset.samples[i][0]
            sperm_name = name.split('/')[-1]
            elements = [sperm_name,labels.numpy()[0]] + features
            all_features.append(elements)

        cols = ["sample","class"]
        for i in range(512):
            f = 'feature'+str(i+1)
            cols.append(f)
        df = pd.DataFrame(all_features, columns=cols)

    return df


def write_predictions( names, original, predictions, fuerza, path, n ):    
    file= open(path+"/predicted_values"+str(n)+".txt","w+")
    for i in range(len(names)):
        name = names[i]
        orig = original[i]
        pred = predictions[i]
        fuerz = max(fuerza[i])
        file.write(name+' ')
        file.write(str(orig)+' ')
        file.write(str(pred)+' ')
        file.write(str(fuerz)+' ')
        file.write( '\n' )
    file.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("-aug", "--augmentation", dest="augmentation", type=str, help="Choose an augmentation method", choices=['Randaugment', 'Gridmask','Smoothmix','Classic'])
    args = parser.parse_args()
    
    main_path = os.path.dirname(os.path.realpath(__file__))
    better_data = main_path+'/BetterResults'

    for aug in [args.augmentation]:
        aug_dir = better_data+'/'+aug
        workdir = os.listdir(aug_dir)
        if '.DS_Store' in workdir:
            workdir.remove('.DS_Store')
        for experiment_data in workdir:
            experiment_number = int(experiment_data.split('-data')[0].replace('experiment',''))
            data_dir = aug_dir+'/'+experiment_data
            resnet_dir = main_path+'/experiments/experiment'+str(experiment_number)+'/resnet34.pth'
            resnet34 = models.resnet34()
            resnet34.load_state_dict(torch.load(resnet_dir))
            resnet34.to('cuda')

            learning_rate = 0.00001
            batch_size = 4
            epochs = 100
            image_dimention = (35,35)
            optimizer = torch.optim.Adam(resnet34.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            trainloader, validloader, testloader = load_data_pytorch_format( data_dir, batch_size, image_dimention)
            _, _, _, _ = process_net(resnet34, epochs, trainloader, validloader, testloader, optimizer, criterion, main_path)

            net = copy.deepcopy(resnet34)
            
            modules=list(net.children())[:-1]
            net=nn.Sequential(*modules)
            for p in net.parameters():
                p.requires_grad = False

            loader_train, loader_test = load_data(data_dir, 1, (35,35))

            features_train = get_features(net,loader_train,'cuda' )
            features_test = get_features(net,loader_test,'cuda' )

            X = features_train.drop(["class"], axis=1)
            X_train = X.drop(["sample"], axis=1)
            y_train = features_train["class"]

            X = features_test.drop(["class"], axis=1)
            X_test = X.drop(["sample"], axis=1)
            y_test = features_test["class"]

            svc = SVM(kernel='poly',gamma='auto', class_weight='balanced', probability=True)
            svc.fit(X_train, y_train)
            names = X['sample']
            predictions = svc.predict(X_test)
            fuerza = svc.predict_proba(X_test)
            print(classification_report(y_test, predictions))
            write_predictions( names, y_test, predictions, fuerza, data_dir, 1)
            
            print('save '+data_dir)


main()
