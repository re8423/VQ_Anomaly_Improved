import torch
import torch.nn as nn
import torch.nn.functional as F

from Pipeline.VQ_anom.blocks import GroupNorm, Swish, ResidualBlock, UpSampleBlock, DownSampleBlock, NonLocalBlock

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layers = [nn.Conv2d(args['image-channels'], 64, 3, 1, 1),
                  
                  ResidualBlock(64, 64),
                  ResidualBlock(64, 64),
                  DownSampleBlock(64),
                  
                  ResidualBlock(64, 64),
                  ResidualBlock(64, 64),
                  DownSampleBlock(64),
                  
                  ResidualBlock(64, 128),
                  ResidualBlock(128, 128),
                  DownSampleBlock(128),
                  
                  ResidualBlock(128, 128),
                  NonLocalBlock(128),
                  ResidualBlock(128, 128),
                  NonLocalBlock(128),
                  DownSampleBlock(128),
                  
                  ResidualBlock(128, 256),
                  ResidualBlock(256, 256),
                  
                  ResidualBlock(256, 256),
                  NonLocalBlock(256),
                  ResidualBlock(256, 256),
                  
                  GroupNorm(256),
                  Swish(),
                  nn.Conv2d(256, args['latent-dim'], 3, 1, 1)
                  ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        layers = [nn.Conv2d(args['latent-dim'], 256, 3, 1, 1),
                  
                  ResidualBlock(256, 256),
                  NonLocalBlock(256),
                  ResidualBlock(256, 256),
                 
                  ResidualBlock(256, 256),
                  NonLocalBlock(256),
                  
                  ResidualBlock(256, 256),
                  NonLocalBlock(256),
                  
                  ResidualBlock(256, 256),
                  NonLocalBlock(256),
                  
                  ResidualBlock(256, 128),
                  NonLocalBlock(128),
                  
                  ResidualBlock(128, 128),
                  NonLocalBlock(128),
                  
                  ResidualBlock(128, 128),
                  NonLocalBlock(128),
                  
                  UpSampleBlock(128),
                  
                  ResidualBlock(128, 128),
                  ResidualBlock(128, 128),
                  ResidualBlock(128, 128),
                  UpSampleBlock(128),
                  
                  ResidualBlock(128, 64),
                  ResidualBlock(64, 64),
                  ResidualBlock(64, 64),
                  UpSampleBlock(64),
                  
                  ResidualBlock(64, 64),
                  ResidualBlock(64, 64),
                  ResidualBlock(64, 64),
                  UpSampleBlock(64),

                  GroupNorm(64),
                  Swish(),
                  nn.Conv2d(64, args['image-channels'], 3, 1, 1)

                 ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
#Vector codebook embedding layer (modified from https://github.com/dome272/VQGAN-pytorch/blob/main/codebook.py)
class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args['num-codebook-vectors']
        self.latent_dim = args['latent-dim']
        self.beta = args['beta']

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        
        mei_reshaped = min_encoding_indices.view(b, h, w)
        
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss, mei_reshaped
    
#VQ-GAN Model (modified from https://github.com/dome272/VQGAN-pytorch/blob/main/vqgan.py)   
class VQVAE(nn.Module):
    def __init__(self, args):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(args).to(device=args['device'])
        self.decoder = Decoder(args).to(device=args['device'])
        self.codebook = Codebook(args).to(device=args['device'])
        self.quant_conv = nn.Conv2d(args['latent-dim'], args['latent-dim'], 1).to(device=args['device'])
        self.post_quant_conv = nn.Conv2d(args['latent-dim'], args['latent-dim'], 1).to(device=args['device'])

    def forward(self, imgs):
        
        c_map, c_indices, q_loss, mei_reshaped = self.encode(imgs)
        out_images = self.decode(c_map)
        return out_images, c_indices, q_loss

    def get_codebook_map(self, imgs):
        encoded = self.encoder(imgs)
        quant_conv_encoded = self.quant_conv(encoded)
        c_map, c_indices, q_loss, mei_reshaped = self.codebook(quant_conv_encoded)
        post_quant_c_map = self.post_quant_conv(c_map)
        return post_quant_c_map
    
    def encode(self, imgs):
        encoded = self.encoder(imgs)
        quant_conv_encoded = self.quant_conv(encoded)
        c_map, c_indices, q_loss, mei_reshaped = self.codebook(quant_conv_encoded)
        return c_map, c_indices, q_loss, mei_reshaped

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        out_images = self.decoder(post_quant_conv_mapping)
        return out_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

        
        