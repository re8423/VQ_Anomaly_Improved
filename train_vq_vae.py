import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from Pipeline.VQ_anom.vq_vae import VQVAE

import lpips

class TrainVQVAE:
    def __init__(self, args, train_loader, val_norm):
                
        self.vqvae = VQVAE(args).to(device=args['device'])
        
        self.opt_vq = torch.optim.Adam(
            list(self.vqvae.encoder.parameters()) +
            list(self.vqvae.decoder.parameters()) +
            list(self.vqvae.codebook.parameters()) +
            list(self.vqvae.quant_conv.parameters()) +
            list(self.vqvae.post_quant_conv.parameters()),
            lr=args['learning-rate'], eps=1e-08, betas=(args['beta1'], args['beta2'])
        )
        
        self.epoch = 0
    
        self.load_pretrained(args, args['resume'])
        
        self.train(args, train_loader, val_norm)
        
    def load_pretrained(self, args, even):
        if even == 'none':
            return
        
        if even == 'even':
            state = torch.load('./vae/' + args['save_even'])
            self.vqvae.load_state_dict(state['vqvae_dict'])
            self.opt_vq.load_state_dict(state['optim'])
            self.epoch = state['epoch']
        else:
            state = torch.load('./vae/'+ args['save_odd'])
            self.vqvae.load_state_dict(state['vqvae_dict'])
            self.opt_vq.load_state_dict(state['optim'])
            self.epoch = state['epoch']+1
        print('loaded model')
        print(self.epoch)
        return

    def train(self, args, train_loader, val_norm):
        train_dataset = train_loader
        steps_per_epoch = len(train_dataset)

        val_counter = 0
        
        for epoch in range(self.epoch, args['epochs']):
            
            avg_train_loss = 0
            
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs, _ = imgs
                    imgs = imgs.to(args['device'])
                    decoded_images, _, q_loss = self.vqvae(imgs)

                    rec_loss = torch.abs(imgs - decoded_images).mean()
    
                    vq_loss = rec_loss + q_loss

                    avg_train_loss = avg_train_loss + vq_loss.tolist()
        
                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)
                
                    self.opt_vq.step()

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                    )
                    pbar.update(0)
                    
            avg_train_loss = avg_train_loss / len(train_dataset)
            
            print(epoch)   
            if epoch%2==0:
                state = {
                    'epoch': epoch,
                    'vqvae_dict': self.vqvae.state_dict(),
                    'optim': self.opt_vq.state_dict(),
                    }
                torch.save(state, os.path.join("vae", args['save_even']))
            elif epoch%2!=0:
                state = {
                    'epoch': epoch,
                    'vqvae_dict': self.vqvae.state_dict(),
                    'optim': self.opt_vq.state_dict(),
                    }
                torch.save(state, os.path.join("vae", args['save_odd']))
                avg_val_loss = 0
                for i, imgs in zip(pbar, val_norm):
                    imgs, _ = imgs
                    imgs = imgs.to(args['device'])
                    decoded_images, _, q_loss = self.vqgan(imgs)
                  
                    rec_loss = torch.abs(imgs - decoded_images).mean()
    
                    vq_loss = perceptual_rec_loss + q_loss
                    avg_val_loss = avg_val_loss + vq_loss.tolist()
                
                avg_val_loss = avg_val_loss / len(val_norm)
                
                if avg_val_loss > avg_train_loss:
                    val_counter = val_counter + 1
            if val_counter ==5:
                print('Halting training via early stopping')
                break
    
