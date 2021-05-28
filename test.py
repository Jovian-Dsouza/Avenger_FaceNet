import os
import torch
import numpy as np
from torchvision import transforms 
from torch import nn
from torch.nn import Softmax
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
from loadOpenFace import prepareOpenFace
from collections import OrderedDict
import argparse

# Check if CUDA GPU is available
useCuda = torch.cuda.is_available()
if useCuda:
    print('CUDA is avialable')
    device = torch.device('cuda:0')
else:
    print('CUDA is not avialable')
    device = torch.device('cpu')

def load_model_from_chk(chk_path):
    '''Returns model and idx_to_class dictionary'''
    try:
        # Load checkpoint 
        checkpoint = torch.load(chk_path)
        idx_to_class = checkpoint['idx_to_class']

        # Load the inception model
        model = prepareOpenFace(useCuda)
        model.eval()
        n_classes = len(idx_to_class)

        # Initialize the classifier model
        classifier_model = nn.Sequential(OrderedDict([
                                            ("nn4_small_v2", model),
                                            ("fc",  nn.Linear(736, n_classes))
                                        ]))

        # load the trained parameters
        classifier_model.load_state_dict(checkpoint['model_state_dict'])
        print("Model Loaded from %s" % chk_path)
        return classifier_model, idx_to_class

    except FileNotFoundError:
        print("Model checkpoint not found %s" % chk_path)
        return None

# Load mtcnn to align and crop images
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)

# tranfomation applied to croped image
face_transform = transforms.Compose([transforms.Resize(96),
                                    transforms.ToTensor()])

softmax = Softmax(dim=1)

# Load the model 
chk_path = 'models/AvengersClassifier.pth'
classifier_model, idx_to_class = load_model_from_chk(chk_path)
classifier_model = classifier_model.to(device)
classifier_model.eval()


def predict(img_path, prob_theshold = 0.9):
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        return 

    # Crop, Align and standardize the Image 
    mtcnn_img = mtcnn(img.convert('RGB'))

    # If no face then return
    if mtcnn_img is None:
        plt.show()
        print("ERROR, Could not detect a face in image")
        return
    
    # Convert to PIL image
    mtcnn_img = Image.fromarray(np.array(mtcnn_img.permute(1, 2, 0).numpy(), dtype=np.uint8))

    # Do the Prediction
    mtcnn_img = face_transform(mtcnn_img).unsqueeze(0)
    mtcnn_img = mtcnn_img.to(device)

    with torch.no_grad():
        label = classifier_model(mtcnn_img)
        label = softmax(label) # To Convert the logit to probabilities

    prob, pred = label.data.max(1, keepdim=True)
    prob, pred = float(prob), int(pred)

    if prob < prob_theshold:
        print("UNKNOWN FACE, but similar to %s with %0.2f%% probability" %
                 (idx_to_class[pred], 100 * prob))
    else:
        print("%s with %0.2f%% probability" %
                 (idx_to_class[pred], 100 * prob))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Takes in image path and does prediction')
    parser.add_argument('-p', '--path', help='Image path')

    args = parser.parse_args()
    img_path = args.path

    print()
    predict(img_path)