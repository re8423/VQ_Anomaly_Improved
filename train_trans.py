import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils

from Pipeline.VQ_anom.transformer import VQGANTransformer

class TrainTransformer:
    def __init__(self, args, train_loader, val_norm):
        self.model = VQGANTransformer(args).to(device=args['device'])
        
        self.epoch = 0
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=arguments['lr'])

        self.load_pretrained(args, args['resume'])
    
        self.train(args, train_loader, val_norm)
        
    def load_pretrained(self, args, even):
        if even == 'none':
            return
        
        if even == 'even':
            state = torch.load('./trans/' + args['save_even'])
            self.model.load_state_dict(state['trans'])
            self.optim.load_state_dict(state['optim'])
            self.epoch = state['epoch']+1
        else:
            state = torch.load('./trans/' + args['save_odd'])
            self.model.load_state_dict(state['trans'])
            self.optim.load_state_dict(state['optim'])
            self.epoch = state['epoch']+1
        print('loaded model')
        print(self.epoch)
        return
    
    def train(self, args, train_loader, val_norm):
        train_dataset = train_loader
        val_counter = 0
        for epoch in range(self.epoch, args['epochs']):
            avg_train_loss = 0
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    imgs, _ = imgs
                    imgs = imgs.to(device=args['device'])
                    logits, targets = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    
                    avg_train_loss = avg_train_loss + torch.mean(loss).tolist()
                    
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
            avg_train_loss = avg_train_loss / len(train_dataset)        
            print(epoch)
            if epoch%2==0:
                state = {
                    'epoch': epoch,
                    'trans': self.model.state_dict(),
                    'optim': self.optim.state_dict(),
                    }
                torch.save(state, os.path.join("trans", args['save_even']))
            elif epoch%2!=0:
                state = {
                    'epoch': epoch,
                    'trans': self.model.state_dict(),
                    'optim': self.optim.state_dict(),
                    }
                torch.save(state, os.path.join("trans", args['save_odd']))
    
                avg_val_loss = 0
                for i, imgs in zip(pbar, val_norm):
                    imgs, _ = imgs
                    imgs = imgs.to(args['device'])
                    logits, targets = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    avg_val_loss = avg_val_loss + torch.mean(loss).tolist()
            
                
                avg_val_loss = avg_val_loss / len(val_norm)
                
                if avg_val_loss > avg_train_loss:
                    val_counter = val_counter + 1
                    
            if val_counter ==5:
                print('Halting training via early stopping')
                break
    
