import os 
import torch
import numpy as np
from torchvision import datasets, transforms, models
import torch.optim as optim
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch.nn as nn
import random
import argparse
from collections import OrderedDict

# Check if CUDA GPU is available
useCuda = torch.cuda.is_available()
if useCuda:
    print('CUDA is avialable')
    device = torch.device('cuda:0')
else:
    print('CUDA is not avialable')
    device = torch.device('cpu')

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

data_dir = 'data/cropped_images'
aligned_data_dir = data_dir + '_aligned'

# dataset = datasets.ImageFolder(aligned_data_dir, transform=transform)
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

batch_size = 64
num_workers = 0 if os.name == 'nt' else 8

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             num_workers=num_workers, shuffle=True)


# Generate triplets
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

from loadOpenFace import prepareOpenFace

model = prepareOpenFace(useCuda)
model.eval()
print("Model Loaded")

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = TripletLoss()

n_epochs = 10

for epoch in tqdm(range(n_epochs), desc="epoch"):
    
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
print("Training Done")





if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(
        description='Takes in image path and does prediction')
    parser.add_argument('-p', '--path', help='Image path')

    args = parser.parse_args()
    print(type(args.path))

