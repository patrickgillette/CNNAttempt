import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import TrainingClasses as tc
from PIL import Image  # Add this import at the top of your file



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_path = "C:/Users/pgillette/source/repos/CNNAttempt/CNNAttempt/scene_detection/seg_train/seg_train/"
    test_path = "C:/Users/pgillette/source/repos/CNNAttempt/CNNAttempt/scene_detection/seg_test/seg_test/"
    
    
    dataset = tc.SceneDetectionDataset(train_path, test_path)
    model_manager = tc.ModelManager()
    
    if not os.path.exists(model_manager.model_path):
        trainer = tc.Trainer(model_manager, dataset)
        trainer.train()
    else:
        print("Pre-trained model exists. Skipping training.")

    pred_path = "C:/Users/pgillette/source/repos/CNNAttempt/CNNAttempt/scene_detection/seg_pred/seg_pred/"
    checkpoint=torch.load('best_checkpoint.model')
    model=tc.ConvNet(num_classes=6)
    model.load_state_dict(checkpoint)
    model.eval()
    
    transformer=transforms.Compose([
        transforms.Resize((150,150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    
    def prediction(img_path,transformer):
        image = Image.open(img_path)
        image_tensor=transformer(image).float()
        image_tensor=image_tensor.unsqueeze_(0)
        if torch.cuda.is_available():
            image_tensor.cuda()
        input=Variable(image_tensor)
        output=model(input)
        index=output.data.numpy().argmax()
        pred=dataset.classes[index]
        return pred


    images_path=glob.glob(pred_path+'/*.jpg')
    pred_dict={}
    for i in images_path:
        pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)
        
    print(pred_dict)


