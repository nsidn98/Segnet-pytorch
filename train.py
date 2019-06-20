"""
Train a SegNet model


Usage:
python train.py --data_root Data --train_path train.txt --img_dir Train --mask_dir Train_annot --save_dir Output
                --checkpoint model_best.pth \
                --gpu 1

Directory structure:
pytorch-segnet
            |_src
                |_ _test.py
                |_ dataset.py
                |_ inference.py
                |_ model.py
                |_ train.py
                |_Data
                    |_ Train
                    |_ Train_annot
                    |_ annotations.txt
                    |_ train.txt
                |_ VOCDevkit
"""

from __future__ import print_function
import argparse
import numpy as np
from dataset import PascalVOCDataset, NUM_CLASSES
from dataset import ScoreDataset, NUM_SCORES
from model import SegNet
import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Arguments
parser = argparse.ArgumentParser(description='Train a SegNet model')

parser.add_argument('--data_root', default="Data")
parser.add_argument('--train_path',default="train.txt")
parser.add_argument('--mask_dir', default="Train_annot")
parser.add_argument('--img_dir', default="Train")
parser.add_argument('--save_dir', default="Output")
parser.add_argument('--custom', default=1, type=int)
parser.add_argument('--checkpoint')
parser.add_argument('--gpu', type=int)

args = parser.parse_args()

if not os.path.exists('./Output'):
    os.makedirs('./Output')

# Constants
NUM_INPUT_CHANNELS = 3

if args.custom:
    NUM_OUTPUT_CHANNELS = NUM_SCORES
else:
    NUM_OUTPUT_CHANNELS = NUM_CLASSES

NUM_EPOCHS = 6000

LEARNING_RATE = 1e-6
MOMENTUM = 0.9
BATCH_SIZE = 10

def train():
    print('TRAINING')
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()
        i=0
        for batch in train_dataloader:
            i+=1
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])
            img = np.transpose(target_tensor.numpy(),(1,2,0))
            # plt.imshow(tar)
            if args.custom:
                target_tensor = target_tensor + 10
            
            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            predicted_tensor, softmaxed_tensor = model(input_tensor)

            optimizer.zero_grad()
            loss = criterion(softmaxed_tensor, target_tensor)
            loss.backward()
            optimizer.step()


            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()

        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            print('SAVING')
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pth"))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))


if __name__ == "__main__":
    data_root = args.data_root
    train_path = os.path.join(data_root, args.train_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu
    print('GPU',GPU_ID)

    print('DATASET')
    if args.custom:
        train_dataset = ScoreDataset(list_file=train_path,
                                     img_dir=img_dir,
                                     mask_dir=mask_dir)
    else:
        train_dataset = PascalVOCDataset(list_file=train_path,
                                        img_dir=img_dir,
                                        mask_dir=mask_dir)
    print('LOADER')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)


    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)

        class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
    else:
        print('MODEL')
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)
        print('STATE_DICT')
        class_weights = 1.0/train_dataset.get_class_probability()
        print('class_weights',len(class_weights))
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


    if args.checkpoint:
        if GPU_ID is None:
            model.load_state_dict(torch.load(args.checkpoint,map_location='cpu'))
        else:
            model.load_state_dict(torch.load(args.checkpoint))


    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)


    train()
