import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score

ROOT_DIR = '/gscratch/ubicomp/hughsun/HAM10000/ResNet'

def getMaleAndFemaleData():
    pass

def evaluate():
    pass

# load the last checkpoint with the best model
device = 'cuda'
model = torchvision.models.resnet50(pretrained=True).to(device)
model.fc = nn.Linear(2048, 7).to(device)
model.load_state_dict(torch.load(f'{ROOT_DIR}/skinmodel50.pt'))

# create validation data loader
valdir = f'{ROOT_DIR}/skin/validation/'

val_transforms = transforms.Compose([
    transforms.Resize((224, 280)),
    torchvision.transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

val_dataset = datasets.ImageFolder(
    valdir, transform=val_transforms)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=False,
    pin_memory=False, drop_last=False)

"""##
## __5. Results.__
### Model stopped training after 290 epochs (around 11-12 hours on 1 Tesla P100 GPU) with Early stopping point at 240 epochs. 
### Result metrics are:
#### Train Accuracy: 0.9715385903231207
#### Val Accuracy: 0.9846359385437542
#### Training Loss: 0.0814
#### Validation Loss: 0.0366
####
### Now we will look at confusion matrix and classification report with Precision, Recall, F1 score and AUC for each class.
"""

num_classes = 7

predlist = torch.zeros(0,dtype=torch.long, device='cpu')
lbllist = torch.zeros(0,dtype=torch.long, device='cpu')
predlistauc = torch.zeros(0,dtype=torch.long, device='cpu')

model.eval()

with torch.no_grad():
    for i, (inputs, classes) in enumerate(val_loader):        
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Append batch prediction results
        predlist=torch.cat([predlist,preds.view(-1).cpu()])
        lbllist=torch.cat([lbllist,classes.view(-1).cpu()])
        predlistauc = torch.cat([predlistauc,nn.functional.softmax(outputs, dim=1).cpu()])
predlist = predlist.numpy()
lbllist = lbllist.numpy()
predlistauc = predlistauc.numpy()

# Confusion matrix, classification report and AUC
conf_mat=confusion_matrix(lbllist, predlist)
target_names = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC',]
ConfusionMatrixDisplay(conf_mat, display_labels=target_names).plot(values_format="d")
print(classification_report(lbllist, predlist, target_names=target_names))
lbllist_one_hot = nn.functional.one_hot(torch.tensor([lbllist]), num_classes=num_classes)
every_auc = roc_auc_score(lbllist_one_hot.view([predlistauc.shape[0], predlistauc.shape[1]]), 
                                          predlistauc, multi_class='ovr', average=None)
for i, every in enumerate(target_names):
    print(f'AUC of class {every} = {every_auc[i]}')

"""##
### For better validation robustness lets spin validation dataset 5 times with random rotations and horizontal flip and then check our metrics again.
"""

val_transforms = transforms.Compose([
    transforms.Resize((224, 280)),
    torchvision.transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(transforms=[transforms.RandomRotation(degrees=(-180, 180))], p=0.99),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


val_dataset = datasets.ImageFolder(
    valdir, transform=val_transforms)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=True,
    pin_memory=False, drop_last=False)

num_classes = 7

predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
predlistauc = torch.zeros(0,dtype=torch.long, device='cpu')
for n in range(5):
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(val_loader):
            model.eval()
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,classes.view(-1).cpu()])
            predlistauc = torch.cat([predlistauc,nn.functional.softmax(outputs, dim=1).cpu()])
predlist = predlist.numpy()
lbllist = lbllist.numpy()
predlistauc = predlistauc.numpy() 

# Confusion matrix, classification report and AUC
conf_mat=confusion_matrix(lbllist, predlist)
target_names = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC',]
ConfusionMatrixDisplay(conf_mat, display_labels=target_names).plot(values_format="d")
print(classification_report(lbllist, predlist, target_names=target_names))
lbllist_one_hot = nn.functional.one_hot(torch.tensor([lbllist]), num_classes=num_classes)
every_auc = roc_auc_score(lbllist_one_hot.view([predlistauc.shape[0], predlistauc.shape[1]]), 
                                          predlistauc, multi_class='ovr', average=None)
for i, every in enumerate(target_names):
    print(f'AUC of class {every} = {every_auc[i]}')

"""## In the end we were able to achieve 97% average F1 score with all by class AUCs close to 100%! 
## And what is interesting, even on 2 classes with smallest representations (only 78 and 58 training images) we still were able to achieve 100 and 99% F1 score respectively.
## I believe that the most important reason for that good of a result were well chosen data augmentations that helped not only to enrich our dataset, but also prevent overfitting. 
"""