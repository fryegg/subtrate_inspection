#!/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from SSIM_PIL import compare_ssim as ssim
import argparse
import pytorch_ssim
from dataset.mvtec_dataset import MVTecDataset
from torch.utils.data import DataLoader
from model.skip_conv import skip_conv
from utils.ms_ssim import MS_SSIM
from torchvision.utils import save_image
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./') # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--class_name', default='4_origin')
    parser.add_argument("--epochs", type=int, default=10, help='Number of times to iterate the whole dataset')
    parser.add_argument("--lr", type=float, default=0.001, help='Learning rate')
    parser.add_argument("--batch_size", default=8, type=int, help='Batch size')
    parser.add_argument("--load", type=str, default='', help='Load pretrained weights')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    # use CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MVTecDataset(dataset_path = os.path.join(args.dataset_path), class_name = args.class_name, is_train=True, resize=400, cropsize=400)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
    test_dataset = MVTecDataset(dataset_path = os.path.join(args.dataset_path), class_name = args.class_name, is_train=False, resize=400, cropsize=400)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    model = skip_conv().to(device)

    # load previous weights (if any)
    #model = torch.load('./save/gh.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ssim_loss = pytorch_ssim.SSIM()
    mse_loss = nn.MSELoss()
    ms_ssim_loss = MS_SSIM(max_val = 1)
    #train
    if args.train:
        model.train()
        for e in tqdm(range(args.epochs)):
            epoch_loss = []
            for batch, (x, _, _) in enumerate(tqdm(train_dataloader, '| feature extraction | train | %s |' % args.class_name)):
                x = x.to(device)
                
                optimizer.zero_grad()

                out = model(x)
                #ssim
                loss = 1-ssim_loss(out, x) + mse_loss(out, x)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
            writer.add_scalar('loss', np.mean(epoch_loss), e)
        torch.save(model, './save/gh3.pth')
    
    else:
        model = torch.load('./save/gh2.pth')
        epoch_loss = []
        model.eval()
        with torch.no_grad():
            for batch, (x, _, _) in enumerate(tqdm(test_dataloader, '| feature extraction | test | %s |' % args.class_name)):
            
                x = x.to(device)
                out = model(x)
                #ssim
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(out, x)
                epoch_loss.append(loss.item())
                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.255]
                )
                out_norm = inv_normalize(out)
                in_norm = inv_normalize(x)
                print(out_norm.shape)
                pred_img = out_norm.view(x.size(0), 3, 400, 400)
                ori_img = in_norm.view(x.size(0), 3, 400, 400)
                stack_img = torch.cat((pred_img, ori_img),2)
                save_image(stack_img,
                '{}/sample_{}.png'.format('./results2',batch))
            print(np.mean(epoch_loss))
            
if __name__ == "__main__":
    main()
