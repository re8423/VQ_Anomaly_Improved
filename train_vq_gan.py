import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from lpips import LPIPS
from Pipeline.VQ_anom.vq_gan import VQGAN, Discriminator

import lpips

class TrainVQGAN:
    def __init__(self, args, train_loader, val_norm):
                
        self.vqgan = VQGAN(args).to(device=args['device'])

        self.discriminator = Discriminator(args).to(device=args['device'])
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args['device'])
        self.perceptual_loss = lpips.LPIPS(net="vgg").to(device=args['device'])
        
        self.opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=args['learning-rate'], eps=1e-08, betas=(args['beta1'], args['beta2'])
        )
        
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=args['learning-rate'], eps=1e-08, betas=(args['beta1'], args['beta2']))
        
        self.epoch = 0
    
        self.load_pretrained(args, args['resume'])
        
        self.train(args, train_loader, val_norm)
        
    def load_pretrained(self, args, even):
        if even == 'none':
            return
        
        if even == 'even':
            state = torch.load('./gan/' + args['save_even'])
            self.vqgan.load_state_dict(state['vqgan_dict'])
            self.opt_vq.load_state_dict(state['optim'])
            self.opt_disc.load_state_dict(state['optim_d'])
            self.epoch = state['epoch']
        else:
            state = torch.load('./gan/'+ args['save_odd'])
            self.vqgan.load_state_dict(state['vqgan_dict'])
            self.opt_vq.load_state_dict(state['optim'])
            self.opt_disc.load_state_dict(state['optim_d'])
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
                    decoded_images, _, q_loss = self.vqgan(imgs)
                    
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(1, epoch*496+i, threshold=0)
                    
                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    
                    perceptual_rec_loss = args['perceptual-loss-factor'] * perceptual_loss + args['rec-loss-factor'] * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()

                    g_loss = -torch.mean(disc_fake)
            
                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
    
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    avg_train_loss = avg_train_loss + vq_loss.tolist()
        
                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)
        
                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        gan_loss = np.round(gan_loss.cpu().detach().numpy().item(), 5)
                    )
                    pbar.update(0)
                    
            avg_train_loss = avg_train_loss / len(train_dataset)
            
            print(epoch)   
            if epoch%2==0:
                state = {
                    'epoch': epoch,
                    'vqgan_dict': self.vqgan.state_dict(),
                    'optim': self.opt_vq.state_dict(),
                    'optim_d': self.opt_disc.state_dict(),
                    }
                torch.save(state, os.path.join("gan", args['save_even']))
            elif epoch%2!=0:
                state = {
                    'epoch': epoch,
                    'vqgan_dict': self.vqgan.state_dict(),
                    'optim': self.opt_vq.state_dict(),
                    'optim_d': self.opt_disc.state_dict(),
                    }
                torch.save(state, os.path.join("gan", args['save_odd']))
                avg_val_loss = 0
                for i, imgs in zip(pbar, val_norm):
                    imgs, _ = imgs
                    imgs = imgs.to(args['device'])
                    decoded_images, _, q_loss = self.vqgan(imgs)
                    
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(1, epoch*496+i, threshold=0)
                    
                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    
                    perceptual_rec_loss = args['perceptual-loss-factor'] * perceptual_loss + args['rec-loss-factor'] * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()

                    g_loss = -torch.mean(disc_fake)
            
                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
    
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss
                    avg_val_loss = avg_val_loss + vq_loss.tolist()
            
                
                avg_val_loss = avg_val_loss / len(val_norm)
                
                if avg_val_loss > avg_train_loss:
                    val_counter = val_counter + 1
            if val_counter ==5:
                print('Halting training via early stopping')
                break
    
