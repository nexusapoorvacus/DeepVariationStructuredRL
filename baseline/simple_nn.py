from PIL import Image
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from tools import union
from models import DetectEdgeNN

import argparse
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import math
import time
import datetime

class BBoxDataset(Dataset):
    """Custom Dataset to iterate through the images.
       (Original version of class written by Ranjay Krishna)
    """
    
    def __init__(self, bboxes, image_dir, image_size):
        """Constructor for BBoxDataset.

        Args:
            bboxes: List of relationships containing the file name, the
                subject bounding box, the object bounding box, and the 
                label for the sample.
            image_dir: Location of where all the images are.
            image_size: The input size of the images to the model.
        """
        self.bboxes = bboxes
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        """Outputs the length of the dataset.

        Returns:
            The length (int) of the dataset.
        """
        return len(self.bboxes)

    def __getitem__(self, index):
        """Loads an image and crops out the union of the two boxes involved
        in the dataset.

        Args:
            index: The image index to retrieve.

        Returns:
            A tensor representation of the union of the boxes involved.
        """
        im, subject_box, object_box, label = self.bboxes[index]
        image = Image.open(self.image_dir /  im)
        image = np.asarray(image)
        if image.ndim == 2:
            image = image.reshape((1, image.shape[0], image.shape[1]))
            image = np.repeat(image, 3, axis=0)
            image = np.transpose(image, (1, 2, 0))
      
        # make sure box dimensions are within image
        subj = self.check_box_dims(subject_box, image.shape)
        obj = self.check_box_dims(object_box, image.shape)
        
        assert subj and obj

        # tranforming image
        im = Image.fromarray(image)
        im = self.transform(im).unsqueeze(0)

        # creating union
        ubox = union(subj, obj)
        crop_union = image[ubox[0]:ubox[1], ubox[2]:ubox[3], :]
        crop_union = Image.fromarray(crop_union)
        crop_union = self.transform(crop_union).unsqueeze(0)

        # cropping subject
        crop_subject = image[subj[0]:subj[1], subj[2]:subj[3], :]
        crop_subject = Image.fromarray(crop_subject)
        crop_subject = self.transform(crop_subject).unsqueeze(0)
        
        # cropping object
        crop_object = image[obj[0]:obj[1], obj[2]:obj[3], :]
        crop_object = Image.fromarray(crop_object) 
        crop_object = self.transform(crop_object).unsqueeze(0)
        return (im, crop_subject, crop_object, crop_union, label)

    def check_box_dims(self, box, image_size):
      if box[0] > image_size[0]-1 or box[2] > image_size[1]-1:
        print("Bad box - sample removed: " + str(box) + ", image size: "  + str(image_size)) 
        return False
      bx = [0, 0, 0, 0]
      bx[0] = min(box[0], image_size[0]-1)
      bx[1] = min(box[1], image_size[0]-1)
      bx[2] = min(box[2], image_size[1]-1)
      bx[3] = min(box[3], image_size[1]-1)
      if (bx[0] == bx[1]) or (bx[2] == bx[3]):
        print("Box has one dimension of size 0: " + str(box))
        return False
      return bx

def make_cuda(*args):
  cuda_args = []
  for arg in args:
    cuda_args.append(arg.cuda())
  return cuda_args

def train(data_loader, model, loss_fn, optimizer, num_epochs, batch_size):
  print("Starting train time: ", datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
  print("Begin Training...")
  model.init_weights()
  model.train()
  print("CUDA Available", torch.cuda.is_available())
  if torch.cuda.is_available():
    model = model.cuda()
  prev_loss = None
  curr_loss = []
  prev_acc = None
  for epoch in range(num_epochs):
    print("Epoch: " + str(epoch + 1))
    for i, data in enumerate(data_loader):
      if len(data) == 0:
        continue
      ims, subjs, objs, unions, labels = data
      var_shape = ims.size()
      ims = torch.autograd.Variable(ims).view(var_shape[0], var_shape[2], var_shape[3], var_shape[4])
      subjs = torch.autograd.Variable(subjs).view(var_shape[0], var_shape[2], var_shape[3], var_shape[4])
      objs = torch.autograd.Variable(objs).view(var_shape[0], var_shape[2], var_shape[3], var_shape[4])
      unions = torch.autograd.Variable(unions).view(var_shape[0], var_shape[2], var_shape[3], var_shape[4])
      labels = torch.autograd.Variable(labels.float())
      optimizer.zero_grad()
      if torch.cuda.is_available():
        ims, subjs, objs, unions, labels = make_cuda(ims, subjs, objs, 
                                                     unions, labels)
      outputs = model(subjs, objs, unions)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()
      curr_loss.append(loss.data[0])
      if (i+1) % 1 == 0:
        print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
          %(epoch+1, num_epochs, i+1, math.ceil(1+len(data_loader.dataset.bboxes)//(batch_size+0.0)), loss.data[0]))
    valid_acc = test(data_loader_validate, model, dataset_name="Validate")
    train_acc = test(data_loader_train, model, dataset_name="Train")
    ave_epoch_loss = sum(curr_loss)/(len(curr_loss)+0.0)
    print("Average Loss: " + str(ave_epoch_loss))
    if prev_acc == None or prev_acc < valid_acc:
      print("Best so far! Saving model...")
      prev_acc = valid_acc 
      prev_loss = ave_epoch_loss
      torch.save(model.state_dict(), args.save_model)
    else:
      for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
        print("Decaying Learning Rate to: " + str(param_group['lr']))
    curr_loss = [] 

def test(data_loader, model, dataset_name="Test"):
    if dataset_name == "Test":
      print("Begin Test on Test Set...")
    elif dataset_name == "Validate":
      print("Begin Test on Validation Set...")
    elif dataset_name == "Train":
      print("Begin Test on Training Set...")
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    if torch.cuda.is_available():
      model = model.cuda()
    correct = 0
    total = 0
    for i, data in enumerate(data_loader):
      if len(data) == 0:
        continue
      ims, subjs, objs, unions, labels = data
      var_shape = ims.size()
      ims = Variable(ims).view(var_shape[0], var_shape[2], var_shape[3], var_shape[4])
      subjs = Variable(subjs).view(var_shape[0], var_shape[2], var_shape[3], var_shape[4])
      objs = Variable(objs).view(var_shape[0], var_shape[2], var_shape[3], var_shape[4])
      unions = Variable(unions).view(var_shape[0], var_shape[2], var_shape[3], var_shape[4])
      labels = torch.LongTensor(labels)
      if torch.cuda.is_available():
        ims, subjs, objs, unions, labels = make_cuda(ims, subjs, objs, 
                                                     unions, labels)
      outputs = model(subjs, objs, unions) 
      predicted = torch.round(outputs.data).long()
      total += labels.size(0)
      correct += (predicted.view(1, -1) == labels).sum()
    acc = (correct + 0.0) / total
    print('Test Accuracy of the '+ dataset_name +' model: %f %%' % acc)
    return acc 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_train', type=str,
                        default='data_train.json',
                        help='Location of file containing the training data samples.')
    parser.add_argument('--bbox_test', type=str,
                        default="data_test.json",
                        help='Location of file containing the test data samples.')
    parser.add_argument('--bbox_validate', type=str,
                        default="data_valid.json",
                        help='Location of the file containing the validation data samples')
    parser.add_argument('--image-dir', type=str, default='/data/apoorvad/VG_100K/',
                         help='Location of the image files')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='# of threads.')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input size for resnet')
    parser.add_argument('--learning-rate', type=float, default=0.00001,
                        help='Learning Rate')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help="Number of epochs to train")
    parser.add_argument('--train', type=str, default='True',
                        help="Train the network.")
    parser.add_argument('--test', type=str, default='True',
                        help="Test the network")
    parser.add_argument('--use-model', type=str, default="use_model.pt",
                        help="Model file to use")
    parser.add_argument('--save-model', type=str, default="saved_model.pt",
                        help="File name to save model in")
    args = parser.parse_args()

    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    print("Learning Rate:", learning_rate)
    print("Batch Size:", batch_size)
    model = DetectEdgeNN()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00005)
    if os.path.isfile(args.use_model):
      print("Loading saved model...")
      model.load_state_dict(torch.load(args.use_model))
    if args.train == 'True':
      bboxes_train = eval("".join(open(args.bbox_train, "r").readlines()))
      bboxes_validate = eval("".join(open(args.bbox_validate, "r").readlines()))
      dataset_train = BBoxDataset(bboxes_train, args.image_dir, args.image_size)
      dataset_validate = BBoxDataset(bboxes_validate, args.image_dir, args.image_size)
      data_loader_train = DataLoader(dataset=dataset_train,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)
      data_loader_validate = DataLoader(dataset=dataset_validate,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)
      train(data_loader_train, model, loss_fn, optimizer, num_epochs, batch_size)
    if args.test == 'True': 
      bboxes_test = eval("".join(open(args.bbox_test, "r").readlines()))
      dataset_test = BBoxDataset(bboxes_test, args.image_dir, args.image_size)
      data_loader_test = DataLoader(dataset=dataset_test,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)
      test(data_loader_test, model)
