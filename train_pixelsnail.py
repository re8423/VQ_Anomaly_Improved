import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import argparse


def TrainPS(vqs, dataloader, val_loader, arguments):
    
    optimizer = torch.optim.Adam(vqs.parameters(), lr=arguments['lr'])
    
    epoch = 0
    
    if arguments['resume'] != 'none':
        state = torch.load('./ps/' + arguments['resume']) #vae_pixel_snail is gan, true_vae_pixel_snail is vae
        vqs.load_state_dict(state['ps_dict'])
        optimizer.load_state_dict(state['optim'])
        print(state['epoch'])
        epoch = state['epoch']
    
    vqs.train()
    train_losses = []
    batch_sizes = []
    val_counter = 0
    for epoch in range(epoch, arguments['max_epoch']):
        avg_train_loss = 0
        with tqdm(range(len(dataloader))) as pbar:
            for x in dataloader:
                x = x[0]
                loss = vqs.loss(x.to(arguments['device']))['loss']

                avg_train_loss = avg_train_loss + loss.item()
                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_value_(vqs.parameters(), 1.)

                optimizer.step()
                train_losses.append(loss.item() * x.shape[0])
                batch_sizes.append(x.shape[0])

                pbar.set_postfix(
                    loss=np.round(loss.item(), 5),
                )
                pbar.update(1)
        avg_train_loss = avg_train_loss / len(dataloader) 
        state = {
            'epoch': epoch,
            'ps_dict': vqs.state_dict(),
            'optim': optimizer.state_dict(),
            }
        torch.save(state, os.path.join("ps", arguments['save_name']))
        if epoch %2!=0:
            avg_val_loss = 0
            for x in val_loader:
                x = x[0]
                loss = vqs.loss(x.to(arguments['device']))['loss']
                avg_val_loss = avg_val_loss + loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)
            
            if avg_val_loss > avg_train_loss:
                    val_counter = val_counter + 1
        if val_counter ==5:
            print('Halting training via early stopping')
            break
        
    return vqs