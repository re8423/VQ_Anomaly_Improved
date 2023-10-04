import torch
import torch.nn as nn
import torch.nn.functional as F

from Pipeline.VQ_anom.vq_gan import VQGAN
from Pipeline.VQ_anom.mingpt import GPT

#modified from https://github.com/dome272/VQGAN-pytorch/blob/main/transformer.py
class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args['sos-token']

        self.vqgan = self.load_vqgan(args)
        
        self.codebook_size = args['num-codebook-vectors']

        transformer_config = {
            "vocab_size": args['num-codebook-vectors'],
            "block_size": 256,
            "n_layer": 24,
            "n_head": 8,
            "n_embd": 1024
        }
        self.transformer = GPT(**transformer_config)

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        state = torch.load(args['vq_model'])
        model.load_state_dict(state['vqgan_dict'])
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=8, p2=8):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    
    def forward_codes(self, indices):
        b, c1, c2 = indices.shape
        
        indices = indices.view(-1, c1*c2)
    
        sos_tokens = torch.ones(indices.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")
        start_indices = indices[:, :indices.shape[1] // 2]
        random_indices = torch.randint_like(start_indices, self.transformer.config.vocab_size)
        
        new_indices = torch.cat((start_indices, random_indices), dim=1)
        new_indices = torch.cat((sos_tokens, new_indices), dim=1)        
        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])
        logits = logits.permute(0, 2, 1)
        logits = logits.view(b, self.codebook_size, c1, c2,)
        
        return logits
    
    def forward(self, x):
        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")
        start_indices = indices[:, :indices.shape[1] // 2]
        random_indices = torch.randint_like(start_indices, self.transformer.config.vocab_size)
        new_indices = torch.cat((start_indices, random_indices), dim=1)
        new_indices = torch.cat((sos_tokens, new_indices), dim=1)    
        target = indices
        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

  