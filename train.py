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

ROOT_DIR = '/gscratch/ubicomp/hughsun/HAM10000/ResNet'

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
        torch.save(model.state_dict(), f'{ROOT_DIR}/skin_models/skinmodel50.pt')
        self.val_loss_min = val_loss

"""### Even though dataset is greatly skewed, during project development I tried to train model with class weights, but I got worse results than with straight up imbalanced training. 
### I came to conclusion that inherent regularization of ResNet50 due to Bottleneck layers and Batch normalization together with weight decay and random data augmentations is enough to prevent model from overfitting even on this imbalanced of a dataset.
"""

traindir = f'{ROOT_DIR}/skin/training/'
valdir = f'{ROOT_DIR}/skin/validation/'

"""##
## __3. Data augmentation.__
### Dataset consists of 7470 unique skin neoplasm photos with height=450 and width=600 pixels.
### By examining the pictures I noticed that in overwhelming majority off them the skin neoplasm is centerd and lies in the center 450x450 crop of the picture (with little exceptions).
### So I believe that there is no reason to squeeze photos on width axis (and lose valuable information) during resize, since our object of interest is almost always in the center crop. We will squueze photos only a little bit to better capture objects that cant fit in 450x450 crop.
### Since we are using ResNet50, to get to 224x224 size, we will first resize to 224x280 (6.25% squeeze on width) and then center crop 224x224 piece.
### Other beneficial transformations could be Random rotation and Horizontal/Vertical flipping. Since skin neoplasms can be in a lot of different shapes, by rotating and horizontally flipping them we will still get skin neoplasms that we would expect our model to classify correctly. These two transformations will help our model greatly. They will inflate our dataset, increase model robustness (especially since we have some classes with very small representation), help with better generalization and overfitting prevention.
### And because we will be using pretrained ResNet50 ImageNet weights to start with, we will use ImageNet style normalization. (only after this project was made I learned that ImageNet normalization only ok if your data set is similar to ImageNet, so I should have calculated mean and std of my own data, but model was able to learn pretty well anyways, so it didnt turn out to be a problem)
### Batch size will be 64.
"""

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

train_dataset = datasets.ImageFolder(
    traindir, transform=train_transforms)

val_dataset = datasets.ImageFolder(
    valdir, transform=val_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True,
    pin_memory=False, drop_last=False)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=True,
    pin_memory=False, drop_last=False)

print(val_dataset.class_to_idx)
print(train_dataset.class_to_idx)

"""##
## __4. Training.__
### Next we define model, change final layer to a 7 way linear classifier, choose Cross Entropy loss function, Adam optimizer (with 0.0001 weight decay), define TensorBoard writer, and since we will be using early stopping, the epoch number will be set to 999.
"""

device = 'cuda'

model = torchvision.models.resnet50(pretrained=True).to(device)

model.fc = nn.Linear(2048, 7).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)

epochs = 999

writer = SummaryWriter(log_dir=f'{ROOT_DIR}/skin_logs/', filename_suffix="skin50")

# Shows info about model
summary(model, input_size=(3, 224, 224))

"""##
### Early stopping tracks validation loss decrease and patience was chosen to be 50. This big of a number is justified, because while developing this project I trained the model a few times and after 80-100 epochs model still slowly and steadily decreases training loss and not overfitting, but validation loss moves pretty stochastically, so not to stop too early patience was chosen to be 50 epochs.
"""

early_stopping = EarlyStopping(patience=50, verbose=True)

for epoch in range(epochs):
    train_loss = 0.00
    val_loss = 0.00
    train_accuracy = Accuracy()
    val_accuracy = Accuracy()
    print(f'Epoch {epoch+1}')

    # Training loop
    for idx, (inputs, labels) in enumerate(Bar(train_loader)):
        model.train()
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
    with torch.no_grad():
        for inputs, labels in val_loader:
            model.eval()           
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