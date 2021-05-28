import os 
import torch
import numpy as np
from torchvision import datasets, transforms, models
import torch.optim as optim
import facenet_pytorch
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch.nn as nn
import random
import argparse
from collections import OrderedDict
from loadOpenFace import prepareOpenFace

### PARAMETERS
data_dir = 'dataset/cropped_images'
n_epochs = 10
chk_path = 'models/AvengersClassifier.pth' # Default checkpoint path
###

### Parse Arguments
parser = argparse.ArgumentParser(
    description='Trains the model and saves the model')

parser.add_argument('-p', '--path', default=chk_path, help='Checkpoint path')
parser.add_argument('-d', '--dataset', default=data_dir, help='Dataset path')
parser.add_argument('-e', '--epochs', type=int, default=n_epochs, help='Number of Epochs')

args = parser.parse_args()
chk_path = args.path
n_epochs = args.epochs
data_dir = args.dataset

### Check if CUDA GPU is available
useCuda = torch.cuda.is_available()
if useCuda:
    print('CUDA is avialable')
    device = torch.device('cuda:0')
else:
    print('CUDA is not avialable')
    device = torch.device('cpu')

### Use MTCNN to Crop and Align Images
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

aligned_data_dir = data_dir + '_aligned'
dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

# Replace the class label with the new path for storing aligned data
dataset.samples = [(p, p.replace(data_dir, aligned_data_dir)) for p, _ in dataset.samples]

batch_size = 32
num_workers = 0 if os.name == 'nt' else 8

dataloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        collate_fn=facenet_pytorch.training.collate_pil)

# Run MTCNN for all the images and save them in new directory
for i, (image, path) in enumerate(tqdm(dataloader, desc="Converting")):
    mtcnn(image, save_path=path)

# Delete to save memory
del mtcnn
del dataloader

print()

#### Augmenting the Dataset
class AugmentDataset(datasets.ImageFolder):
    def __init__(self, root, transform = None):
        super().__init__(root, transform)
        self.all_labels = [int(x[1]) for x in self.imgs]
        
        self.horizontalTransform = transforms.RandomHorizontalFlip(1)
    
    def __len__(self):
        return 2 * super().__len__()
    
    def __getitem__(self, item):
        if item < super().__len__():
            image, label = super().__getitem__(item)
        else:
            item -= super().__len__()
            image, label = super().__getitem__(item)
            image = self.horizontalTransform(image)
             
        return image, label

transform = transforms.Compose([transforms.Resize(96),
                                transforms.ToTensor()])

dataset = AugmentDataset(aligned_data_dir, transform=transform)
idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

total_count = len(dataset)
train_count = int(0.8 * total_count)
test_count = total_count - train_count

train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                           [train_count, test_count])
print('Total Images : ', total_count)
print('Num of Train Images : ', len(train_dataset))
print('Num of Test Images : ', len(test_dataset))
print()

batch_size = 64
num_workers = 0 if os.name == 'nt' else 8

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=True)


### Generate triplets Function
def generate_triplets(images, labels):
    positive_images = []
    negative_images = []
    batch_size = len(labels)
    
    for i in range(batch_size):
        anchor_label = labels[i]

        positive_list = []
        negative_list = []

        for j in range(batch_size):
            if j != i:
                if labels[j] == anchor_label:
                    positive_list.append(j)
                else:
                    negative_list.append(j)

        positive_images.append(images[random.choice(positive_list)])
        negative_images.append(images[random.choice(negative_list)])

    positive_images = torch.stack(positive_images)
    negative_images = torch.stack(negative_images)
    
    return positive_images, negative_images

### Define Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
    
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor, positive, negative): # (batch_size , emb_size)
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.alpha)
        return losses.mean()

# Load inception model
model = prepareOpenFace(useCuda)
model.eval()
print("Inception Model Loaded")

# Define optimizer and loss for inception model
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = TripletLoss()

# Training the inception model
for epoch in range(n_epochs):
    train_loss = 0 
    count = 0
    
    ## Training Loop
    model.train()
    for batch, (images, labels) in enumerate(tqdm(train_dataloader, \
                                             desc="Training", leave=False)):
        
        positives , negatives = generate_triplets(images, labels)
        
        # Move tensor to device
        images, labels = images.to(device), labels.to(device)
        positives, negatives = positives.to(device), negatives.to(device) 
        
        optimizer.zero_grad()
        
        # Seaseme Network
        anchor_out = model(images)
        positive_out = model(positives)
        negative_out = model(negatives)
        
        # Get the loss
        loss = loss_fn(anchor_out, positive_out, negative_out)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.detach().item()
        count =  len(labels)
        

    print('Epoch : %d/%d - Loss: %0.4f' % 
          (epoch+1, n_epochs, train_loss / count))
    train_loss = 0.0

model.eval()
print("Inception Model : Training Done\n")

### Transfer Learning the classifier
n_classes = len(dataset.class_to_idx)

# Define the classifier model
classifier_model = nn.Sequential(OrderedDict([
                                    ("nn4_small_v2", model),
                                    ("fc",  nn.Linear(736, n_classes))
                                ]))

classifier_model = classifier_model.to(device)

# Freeze the parameters in the nn4_small_v2 layer
for param in classifier_model.parameters():
    param.requires_grad = False

for param in classifier_model.fc.parameters():
    param.requires_grad = True
    
# Define optimizer and loss for classifier model
optimizer = optim.Adam(classifier_model.fc.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

### Training the Classifier
print("Training Classifier")
def train(n_epochs, dataloader, model, optimizer, loss_fn):
    '''returns Trained classifier model'''
    
    for epoch in range(n_epochs):

        train_loss = 0.0
        count = 0
        
        # Training loop 
        model.train()
        for batch, (images, labels) in enumerate(tqdm(dataloader, \
                                                 desc="Training", leave=False)):
            
            # Move Tensor to appropriate device
            images, labels = images.to(device), labels.to(device)
 
            optimizer.zero_grad()

            out = model(images)
            
            # Get the loss
            loss = loss_fn(out, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.detach().item()
            count =  len(labels)


        print('Epoch : %d/%d - Loss: %0.4f' % 
              (epoch+1, n_epochs, train_loss / count))
        train_loss = 0.0

    model.eval()
    print("Classifier Model : Training Done\n")
    return model

# call the train function
classifier_model = train(10 , train_dataloader, classifier_model, optimizer, loss_fn)

### Testing the classifier
def test(dataloader, model, loss_fn):
    
    test_loss = 0.0
    total = 0
    correct = 0

    # Testing loop 
    model.eval()
    for batch, (images, labels) in enumerate(tqdm(dataloader, \
                                             desc="Testing")):

        # Move Tensor to appropriate device
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            out = model(images)

        loss = loss_fn(out, labels)
        test_loss += loss.detach().item()
        
        # Get the class with max probability
        pred = out.data.max(1, keepdim=True)[1]
        # Compare predictions with true label
        correct += np.sum(np.squeeze(pred.eq(labels.view_as(pred))).cpu().numpy())
        total += labels.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss/total))
    print('Test Accuracy : %d%% (%d/%d)' % (
            100 * correct / total, correct, total))
    print()
    
    return(float(correct / total))

# call the test function
current_accuracy = test(test_dataloader, classifier_model, loss_fn)
    
### Define Function to save model
def save_model(model, chk_path, idx_to_class, current_accuracy=1.0):
    '''Saves the model only if model doesnt exist or
       if the previous model accuracy was better'''
    try:
        checkpoint = torch.load(chk_path, map_location=torch.device('cpu'))
        if(current_accuracy < checkpoint['accuracy']):
            print("Not Saving, Previous model was better")
            return
        
    except FileNotFoundError:
        print("Previous model not found")
        
    torch.save({
        'model_state_dict' : model.state_dict(),
        'accuracy' : current_accuracy,
        'idx_to_class': idx_to_class
    }, chk_path)

    print("Model Saved : %s" % chk_path)

save_model(classifier_model, chk_path, idx_to_class, current_accuracy)