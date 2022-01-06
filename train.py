import os
import torch
import torchvision
import pandas as pd
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from barbar import Bar
from torchsummary import summary
from ignite.metrics import Accuracy

from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT_DIR = '/gscratch/ubicomp/hughsun/HAM10000/ResNet'
TRAIN_PERC = 0.7 # training set size
VAL_PERC = 0.15 # validation set size
TEST_PERC = 0.15 # test set size

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, seed, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.seed = seed
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when monitored metric decrease.'''
        if self.verbose:
            self.trace_func(f'Monitored metric has improved ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.isdir(f'{ROOT_DIR}/skin_models'):
            os.mkdir(f'{ROOT_DIR}/skin_models')
        seed = self.seed
        torch.save(model.state_dict(), f'{ROOT_DIR}/skin_models/skinmodel50_{seed}.pt')
        self.val_loss_min = val_loss

class SkinDataset(Dataset):
    '''
    create a dataset of skin images from the passed in metadata dataframe,
    and using the passed in transform.
    '''
    def __init__(self, df_metadata, transform):
        self.metadata = df_metadata
        self.transform = transform
        self.classes = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
        self.class_to_idx = {'mel': 0, 'nv': 1, 'bcc': 2, 'akiec': 3, 'bkl': 4, 'df': 5, 'vasc': 6}
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img = self.read_image(idx)
        tensor_img = self.transform(img)
        age = self.metadata.iloc[idx]['age']
        sex = self.metadata.iloc[idx]['sex']
        dx = self.metadata.iloc[idx]['dx']
        dx = self.class_to_idx[dx]
        return (tensor_img, age, sex, dx)
        
    def read_image(self, idx):
        image_file_name = self.metadata.iloc[idx]['image_id']
        return Image.open(f'{ROOT_DIR}/HAM10000/{image_file_name}.jpg')

def train(train_loader, val_loader, seed):
    '''
    train + save model based on a provided seed number + train val loaders.
    '''

    device = 'cuda'
    model = torchvision.models.resnet50(pretrained=True).to(device)
    model.fc = nn.Linear(2048, 7).to(device)
    # Shows info about model
    # summary(model, input_size=(3, 224, 224))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)
    epochs = 999

    writer = SummaryWriter(log_dir=f'{ROOT_DIR}/skin_logs/', filename_suffix=f"skin50_{seed}")
    """
    Early stopping tracks validation loss decrease and patience was chosen to be 50. 
    This big of a number is justified, because while developing this project I trained 
    the model a few times and after 80-100 epochs model still slowly and steadily decreases
    training loss and not overfitting, but validation loss moves pretty stochastically, 
    so not to stop too early patience was chosen to be 50 epochs.
    """
    early_stopping = EarlyStopping(seed=seed, patience=50, verbose=True)
    
    for epoch in range(epochs):
        train_loss = 0.00
        val_loss = 0.00
        train_accuracy = Accuracy()
        val_accuracy = Accuracy()
        print(f'Epoch {epoch+1}')

        # Training loop
        model.train()
        for idx, (inputs, ages, sexes, labels) in enumerate(Bar(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            train_loss += loss.item()
            train_accuracy.update((nn.functional.softmax(outputs, dim=1), labels))
        print(f"Train Accuracy: {train_accuracy.compute()}")
        train_loss /= len(train_loader)
        train_loss_formated = "{:.4f}".format(train_loss)

        # Validation loop
        model.eval() 
        with torch.no_grad():
            for inputs, ages, sexes, labels in val_loader:     
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy.update((nn.functional.softmax(outputs, dim=1), labels))
        print(f"Val Accuracy: {val_accuracy.compute()}")
        val_loss /= len(val_loader)
        val_loss_formated = "{:.4f}".format(val_loss)
        print(f'Training Loss: {train_loss_formated}')
        print(f"Validation Loss: {val_loss_formated}")

        # TensorBoard writer 
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalar('Loss/val', val_loss, epoch+1)
        writer.add_scalar('Accuracy/train', train_accuracy.compute(), epoch+1)
        writer.add_scalar('Accuracy/val', val_accuracy.compute(), epoch+1)

        # Early Stopping
        early_stopping(val_loss, model)       
        if early_stopping.early_stop:
            print("Early stopping")
            break

def main():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 280)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(-180, 180))], p=0.99),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 280)),
        torchvision.transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    metadata = pd.read_csv(f'{ROOT_DIR}/HAM10000_metadata.csv')

    data_size = len(metadata)
    train_size = int(data_size * TRAIN_PERC)
    val_size = int(data_size * VAL_PERC)
    test_size = data_size - train_size - val_size

    seeds = [446, 1234, 1111]
    for seed in seeds:
        shuffled_metadata = metadata.sample(frac = 1, random_state = seed)
        train_metadata = shuffled_metadata[:train_size]
        val_metadata = shuffled_metadata[train_size: train_size + val_size]
        test_metadata = shuffled_metadata[train_size + val_size: train_size + val_size + test_size]

        train_dataset = SkinDataset(train_metadata, train_transforms)
        val_dataset = SkinDataset(val_metadata, val_transforms)

        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True,
            pin_memory=True, drop_last=False) 
        val_loader = DataLoader(
            val_dataset, batch_size=64, shuffle=False,
            pin_memory=True, drop_last=False)
        
        train(train_loader, val_loader, seed)

main()