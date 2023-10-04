from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import Pipeline.VQ_anom.nn_blocks as nn_blocks

#modified from https://github.com/snavalm/lsr_mood_challenge_2020/blob/master/nets_AR.py
class PixelSNAIL(nn.Module):
    def __init__(self,d,
        shape = (64,64),n_channels=64, n_block=4,n_res_block = 2, dropout_p=0.1, downsample = 1, non_linearity = F.elu):
        super().__init__()

        self.non_linearity = non_linearity
        height, width = shape

        self.d = d
        self.ini_conv = nn_blocks.MaskedConv(d,n_channels, kernel_size=7, stride = downsample, mask_type='A',)

        height //= downsample
        width //= downsample

        # Creates a grid with coordinates within image
        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))

        self.blocks = nn.ModuleList()

        for i in range(n_block):
            self.blocks.append(nn_blocks.PixelBlock(n_channels,n_channels, n_res_block=n_res_block, shape = (height,width),
                                          dropout_p=dropout_p,cond_channels=None,non_linearity = non_linearity))

        self.upsample = nn.ConvTranspose2d(n_channels, n_channels,kernel_size=downsample, stride=downsample)

        self.out = nn_blocks.WNConv2d(n_channels, d, 1)

    def forward(self, inp):
        inp = F.one_hot(inp, self.d).permute(0, 3, 1, 2).type_as(self.background)

        out = self.ini_conv(inp)
        
        batch, _, height, width = out.shape
        background = self.background.expand(batch, -1, -1, -1)

        for block in self.blocks:
            out = block(out, background=background)

        out = self.upsample(self.non_linearity(out))
        out = self.out(self.non_linearity(out))
        return out

    def loss(self, x, reduction = 'mean'):
        logits = self.forward(x)
        nll = F.cross_entropy(logits, x,reduction=reduction)
        return OrderedDict(loss=nll)

    def sample(self, n, img_size = (128,128)):
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long().to(device)
        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    logits = self(samples)[:, :, r, c]
                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.cpu().numpy()
    
    
class VQLatentSNAIL(PixelSNAIL):
    def __init__(self, feature_extractor_model, **kwargs):
        super().__init__(d = feature_extractor_model.codebook.num_codebook_vectors,
                                       **kwargs)

        for p in feature_extractor_model.parameters():
            p.requires_grad = False

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor_model.eval()

    def retrieve_codes(self,x):
        with torch.no_grad():
            self.feature_extractor_model.eval()
            _, _, _, code = self.feature_extractor_model.encode(x)
#             _,_,code = self.feature_extractor_model.codebook(z)
        return code

    def forward(self, x):
        # Retrieve codes for images
        code = self.retrieve_codes(x)
        return super(VQLatentSNAIL,self).forward(code)

    def forward_latent(self, code):
        return super(VQLatentSNAIL,self).forward(code)

    def loss(self, x, reduction = 'mean'):
        # Retrieve codes for images
        code = self.retrieve_codes(x)
        logits = super(VQLatentSNAIL,self).forward(code)
        nll = F.cross_entropy(logits, code, reduction = reduction)
        return OrderedDict(loss=nll)
    
    def loss_sample(self, x, reduction = 'none'):
        # Retrieve codes for images
        code = self.retrieve_codes(x)
        logits = super(VQLatentSNAIL,self).forward(code)
        nll = F.cross_entropy(logits, code, reduction = reduction)
        return nll

    def sample(self, n, img_size = (128,128)):
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long()
        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    logits = super(VQLatentSNAIL,self).forward(samples)[:, :, r, c]
                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.cpu().numpy()