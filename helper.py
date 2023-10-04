import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image, make_grid

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score #Note accuracy is computed in many of the functions but not used in the final results (see paper)
from sklearn.metrics import roc_curve, auc

#For preforming Light-Pixel-wise detection (Input: VQ+AR model, dataloader -> Output:anomaly scores based on light-pixel-wise detection, all likelihood losses, all original images, all restorations, all residuals)
def light_pixel_wise(t_model, dataloader, params, txt_name, limit = 50, n=5, threshold_log_p=5):
    print('only testing ' + str(limit) + ' samples')

    txtfile = '/light/' + txt_name
    print('saving data in: ' + txtfile)

    
    pixel_scores = []
    with open(txtfile, 'r') as f:
        for line in f:
            pixel_scores.append(float(line[:-2]))
            
    print('found ' + str(len(pixel_scores)) + ' entries in txtfile, resuming testing from there')
    
    smooth = nn.Sequential(nn.MaxPool3d(kernel_size=3,padding=1,stride=1), #should be minpooling
                           nn.AvgPool3d(kernel_size=(3,7,7),padding=(1,3,3),stride=1),
                          ).to(params['device'])
    all_losses = []
    outs = []
    all_diffs = []
    all_ogs = []
    
    with open(txtfile, 'a') as f:
        for batch_idx, (x, _) in enumerate(tqdm(dataloader)):
            if batch_idx == limit:
                break
            elif batch_idx < len(pixel_scores):
                continue
                
            x = x.to(params['device'])

            if params['ps'] == 'true':
                out, samples, losses = light_restore_ps(t_model, n, x, threshold_log_p)
            else:
                out, samples, losses = light_restore(t_model, n, x, threshold_log_p)

            all_losses.append(losses)
            outs.append(out)

            ogs = x.repeat(n, 1, 1, 1)
            all_ogs.append(ogs)

            diffs = torch.abs(ogs - out)

            sim_imgwise = torch.mean(diffs,(2,3)).unsqueeze(2).unsqueeze(3)
            sim_imgwise = torch.softmax(3/sim_imgwise,1) #k=3

            diffs = (diffs*sim_imgwise).sum(1,keepdims=True)

            diffs = diffs.squeeze().unsqueeze(0).unsqueeze(0)
            smooth_diffs = -smooth(-diffs)
            smooth_diffs = smooth_diffs.squeeze().unsqueeze(1)
            all_diffs.append(smooth_diffs)

            smooth_diffs = smooth_diffs.flatten().clamp_max(1.).cpu() / 1.


            pixel_scores.append(torch.mean(smooth_diffs))
            f.write(f"{torch.mean(smooth_diffs)}\n")

            
    return pixel_scores, all_losses, all_ogs, outs, all_diffs

#For preforming Light-Pixel-wise restoration (Input: VQ+AR model, image -> Output: restorations, latent variable samples, likelihood losses
# modified from https://github.com/snavalm/lsr_mood_challenge_2020/blob/master/1LatentSpatialRestoration-MOODBrainDataset.ipynb
def light_restore(model, n, img, threshold_log_p = 5):
    
    model.eval()
        
    #Use VQ-VAE to encode original image
    _, _, _, codes = model.vqgan.encode(img)
    code_size = codes.shape[-2:]
    
    losses = []
    
    with torch.no_grad():
        samples = codes.clone().repeat(1,n,1,1).reshape(img.shape[0]*n,*code_size)

        all_logits = model.forward_codes(samples)
        
        # Iterate through latent variables. 
        for r in range(code_size[0]):
            for c in range(code_size[1]):

#                 logits = model.forward_codes(samples)[:, :, r, c]
                logits = all_logits[:, :, r, c]
                loss =F.cross_entropy(logits, samples[:, r, c], reduction='none')

                # Replace sample if above threshold
                probs = F.softmax(logits, dim=1)
                losses.append(torch.mean(loss))
                samples[loss > threshold_log_p, r, c] = torch.multinomial(probs, 1).squeeze(-1)[loss > threshold_log_p]
        
        # Retrieve z for the latent codes
        z = model.vqgan.codebook.embedding(samples.unsqueeze(1))
        z = z.squeeze(1).permute(0,3,1,2).contiguous()
        # Decode to pixel space splitting computation in batches
        x_tilde = []
        for i in range(img.shape[0]):
            x_tilde.append(model.vqgan.decode(z[i*n:(i+1)*n]))
        x_tilde = torch.cat(x_tilde)

    return x_tilde.reshape(n, 3,*img.shape[-2:]), samples.reshape(img.shape[0],n,*code_size), losses

#For preforming Light-Pixel-wise restoration for PixelSnail as AR model; function is separated from above as PixelSnail and Transformer implementations take in/produce of differing format(Input: VQ+AR model, image -> Output: restorations, latent variable samples, likelihood losses
# modified from https://github.com/snavalm/lsr_mood_challenge_2020/blob/master/1LatentSpatialRestoration-MOODBrainDataset.ipynb
def light_restore_ps(model, n, img, threshold_log_p = 5):
    
    model.eval()
        
    #Use VQ-VAE to encode original image
    codes = model.retrieve_codes(img)
    code_size = codes.shape[-2:]
    
    losses = []
    
    with torch.no_grad():
        samples = codes.clone().repeat(1,n,1,1).reshape(img.shape[0]*n,*code_size)

        all_logits = model.forward_latent(samples)
        
        # Iterate through latent variables. 
        for r in range(code_size[0]):
            for c in range(code_size[1]):

#                 logits = model.forward_codes(samples)[:, :, r, c]
                logits = all_logits[:, :, r, c]
                loss =F.cross_entropy(logits, samples[:, r, c], reduction='none')

                # Replace sample if above threshold
                probs = F.softmax(logits, dim=1)
                losses.append(torch.mean(loss))
                samples[loss > threshold_log_p, r, c] = torch.multinomial(probs, 1).squeeze(-1)[loss > threshold_log_p]
        
         # Retrieve z for the latent codes
        z = model.feature_extractor_model.codebook.embedding(samples.unsqueeze(1))
        z = z.squeeze(1).permute(0,3,1,2).contiguous()
        # Decode to pixel space splitting computation in batches
        x_tilde = []
        for i in range(img.shape[0]):
            x_tilde.append(model.feature_extractor_model.decode(z[i*n:(i+1)*n]))
        x_tilde = torch.cat(x_tilde)

    return x_tilde.reshape(n, 3,*img.shape[-2:]), samples.reshape(img.shape[0],n,*code_size), losses

#For preforming Pixel-wise detection (Input: VQ+AR model, dataloader -> Output:anomaly scores based on pixel-wise detection, all likelihood losses, all original images, all restorations, all residuals)
def pixel_wise(t_model, dataloader, params, txt_name, limit = 50, n=5, threshold_log_p=5):
    print('only testing ' + str(limit) + ' samples')

    txtfile = '/pixel_wises/' + txt_name
    print('saving data in: ' + txtfile)

    pixel_scores = []
    with open(txtfile, 'r') as f:
        for line in f:
            pixel_scores.append(float(line[:-2]))
            
    print('found ' + str(len(pixel_scores)) + ' entries in txtfile, resuming testing from there')
    
    smooth = nn.Sequential(nn.MaxPool3d(kernel_size=3,padding=1,stride=1), #should be minpooling
                           nn.AvgPool3d(kernel_size=(3,7,7),padding=(1,3,3),stride=1),
                          ).to(params['device'])
    all_losses = []
    outs = []
    all_diffs = []
    all_ogs = []
    
    with open(txtfile, 'a') as f:
        for batch_idx, (x, _) in enumerate(tqdm(dataloader)):
            if batch_idx == limit:
                break
            elif batch_idx < len(pixel_scores):
                continue
                
            x = x.to(params['device'])
            
            if params['ps'] == 'true':
                out, samples, losses = restore_ps(t_model, n, x, threshold_log_p)
            else:
                out, samples, losses = restore(t_model, n, x, threshold_log_p)
                
            all_losses.append(losses)
            outs.append(out)
            
            ogs = x.repeat(5, 1, 1, 1)
            all_ogs.append(ogs)

            diffs = torch.abs(ogs - out)

            sim_imgwise = torch.mean(diffs,(2,3)).unsqueeze(2).unsqueeze(3)
            sim_imgwise = torch.softmax(3/sim_imgwise,1) #k=3

            diffs = (diffs*sim_imgwise).sum(1,keepdims=True)

            diffs = diffs.squeeze().unsqueeze(0).unsqueeze(0)
            smooth_diffs = -smooth(-diffs)
            smooth_diffs = smooth_diffs.squeeze().unsqueeze(1)
            all_diffs.append(smooth_diffs)

            smooth_diffs = smooth_diffs.flatten().clamp_max(1.).cpu() / 1.
            
            
            pixel_scores.append(torch.mean(smooth_diffs))
            
            f.write(f"{torch.mean(smooth_diffs)}\n")
            
    return pixel_scores, all_losses, all_ogs, outs, all_diffs

#For preforming Pixel-wise restoration (Input: VQ+AR model, image -> Output: restorations, latent variable samples, likelihood losses
# modified from https://github.com/snavalm/lsr_mood_challenge_2020/blob/master/1LatentSpatialRestoration-MOODBrainDataset.ipynb
def restore(model, n, img, threshold_log_p = 5):
    
    model.eval()
        
    #Use VQ-VAE to encode original image
    _, _, _, codes = model.vqgan.encode(img)
    code_size = codes.shape[-2:]
    
    losses = []
    
    with torch.no_grad():
        samples = codes.clone().repeat(1,n,1,1).reshape(img.shape[0]*n,*code_size)

        # Iterate through latent variables. 
        for r in range(code_size[0]):
            for c in range(code_size[1]):
                logits = model.forward_codes(samples)[:, :, r, c]
                loss = F.cross_entropy(logits, samples[:, r, c], reduction='none')

                # Replace sample if above threshold
                probs = F.softmax(logits, dim=1)
                losses.append(torch.mean(loss))
                samples[loss > threshold_log_p, r, c] = torch.multinomial(probs, 1).squeeze(-1)[loss > threshold_log_p]
        
        # Retrieve z for the latent codes
        z = model.vqgan.codebook.embedding(samples.unsqueeze(1))
        z = z.squeeze(1).permute(0,3,1,2).contiguous()
        # Decode to pixel space splitting computation in batches
        x_tilde = []
        for i in range(img.shape[0]):
            x_tilde.append(model.vqgan.decode(z[i*n:(i+1)*n]))
        x_tilde = torch.cat(x_tilde)

    return x_tilde.reshape(n, 3,*img.shape[-2:]), samples.reshape(img.shape[0],n,*code_size), losses

#For preforming Pixel-wise restoration for PixelSnail as AR model; function is separated from above as PixelSnail and Transformer implementations take in/produce of differing format(Input: VQ+AR model, image -> Output: restorations, latent variable samples, likelihood losses
# modified from https://github.com/snavalm/lsr_mood_challenge_2020/blob/master/1LatentSpatialRestoration-MOODBrainDataset.ipynb
def restore_ps(model, n, img, threshold_log_p = 5):
    
    model.eval()
        
    #Use VQ-VAE to encode original image
    codes = model.retrieve_codes(img)
    code_size = codes.shape[-2:]
    
    losses = []
    
    with torch.no_grad():
        samples = codes.clone().repeat(1,n,1,1).reshape(img.shape[0]*n,*code_size)

        # Iterate through latent variables. 
        for r in range(code_size[0]):
            for c in range(code_size[1]):
                
                logits = model.forward_latent(samples)[:, :, r, c]
                loss = F.cross_entropy(logits, samples[:, r, c], reduction='none')

                # Replace sample if above threshold
                probs = F.softmax(logits, dim=1)
                losses.append(torch.mean(loss))
                samples[loss > threshold_log_p, r, c] = torch.multinomial(probs, 1).squeeze(-1)[loss > threshold_log_p]
        
        # Retrieve z for the latent codes
        z = model.feature_extractor_model.codebook.embedding(samples.unsqueeze(1))
        z = z.squeeze(1).permute(0,3,1,2).contiguous()
        # Decode to pixel space splitting computation in batches
        x_tilde = []
        for i in range(img.shape[0]):
            x_tilde.append(model.feature_extractor_model.decode(z[i*n:(i+1)*n]))
        x_tilde = torch.cat(x_tilde)

    return x_tilde.reshape(n, 3,*img.shape[-2:]), samples.reshape(img.shape[0],n,*code_size), losses

#Calculate best Lambda_S value given potential Lambda_S values and their F1, accuracy, and AUC scores
def get_best_score_threshold(res):
    percs = []
    aucs = []
    accs = []
    f1s = []

    best_percs = 0
    best_auc = 0
    best_acc = 0
    best_f1 = 0

    for key in res:

        if res[key]['auc'] > best_auc:
            best_percs = key
            best_auc = res[key]['auc']
            best_acc = res[key]['acc']
            best_f1 = res[key]['f1']

        percs.append(key)
        aucs.append(res[key]['auc'])
        accs.append(res[key]['acc'])
        f1s.append(res[key]['f1'])

    print('best perc: ' + str(best_percs))
    print('best f1: ' + str(best_f1))
    print('best acc: ' + str(best_acc))
    print('best auc: ' + str(best_auc))

    plt.plot(percs, f1s, label='f1')
    plt.plot(percs, accs, label='accuracy')
    plt.plot(percs, aucs, label='auc')
    plt.legend(loc="upper right")
    plt.title('f1, accuracy, auc values with different score thresholds')
    plt.show()


#Compute potential Lambda_S values and find their respective F1, accuracy, and AUC score
def get_sample_valid(val_anom_loader, val_norm_loader, t_model, params, thres_interval):
    
    raw_scores = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(val_anom_loader)):
            x = x.to(params['device'])
            logits, targets = t_model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction = 'none').tolist()
            for i in nll:
                raw_scores.append(i)

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(val_norm_loader)):
            x = x.to(params['device'])
            logits, targets = t_model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction = 'none').tolist()
            for i in nll:
                raw_scores.append(i)

    all_scores = raw_scores.copy()
    all_scores = np.array(all_scores)
    print('Percentiles: ')
    print('20: ' + str(np.percentile(all_scores, 20)))
    print('30: ' + str(np.percentile(all_scores, 30)))
    print('40: ' + str(np.percentile(all_scores, 40)))
    print('50: ' + str(np.percentile(all_scores, 50)))
    print('60: ' + str(np.percentile(all_scores, 60)))
    print('70: ' + str(np.percentile(all_scores, 70)))
    print('80: ' + str(np.percentile(all_scores, 80)))
    print('90: ' + str(np.percentile(all_scores, 90)))
    print('98: ' + str(np.percentile(all_scores, 98)))
    
    percs = []
    percs.append(np.percentile(all_scores, 20))
    percs.append(np.percentile(all_scores, 30))
    percs.append(np.percentile(all_scores, 40))
    percs.append(np.percentile(all_scores, 50))
    percs.append(np.percentile(all_scores, 60))
    percs.append(np.percentile(all_scores, 70))
    percs.append(np.percentile(all_scores, 80))
    percs.append(np.percentile(all_scores, 90))
    percs.append(np.percentile(all_scores, 98))
    
    results = {}
    
    for perc in percs:
        print(perc)
        anom_scores = []
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tqdm(val_anom_loader)):
                x = x.to(params['device'])
                logits, targets = t_model(x)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction = 'none').tolist()
                score = 0
                for i in nll:
                    raw_scores.append(i)
                    if i > perc:
                        score = score + i
                anom_scores.append(score)

        norm_scores = []
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tqdm(val_norm_loader)):
                x = x.to(params['device'])
                logits, targets = t_model(x)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction = 'none').tolist()
                score = 0
                for i in nll:
                    raw_scores.append(i)
                    if i > perc:
                        score = score + i
                norm_scores.append(score)
        
        norm = norm_scores
        anom = anom_scores

        df_list = norm+anom
        df = np.array(df_list)

        norm_l = np.array([0] * len(norm_scores)).flatten()
        anom_l = np.array([1] * len(anom_scores)).flatten()
        labels = np.concatenate((norm_l, anom_l), axis=0)

        max_range = max(anom_scores)
        min_range = min(norm_scores)
        thres_range = np.arange (min_range, max_range, thres_interval)
        f1s = []
        accs = []
        aucs = []

        best_thres = 0
        max_f1 = 0
        max_acc = 0
        max_auc = 0

        for thres in tqdm(thres_range):
            preds = []
            for i in df_list:
                if i > thres:
                    preds.append(1)
                else:
                    preds.append(0)
            preds = np.array(preds)

            f1 = f1_score(labels, preds, average='macro')
            f1s.append(f1)
            acc = accuracy_score(labels, preds)
            accs.append(acc)
            fpr, tpr, thresholds = roc_curve(labels, preds)
            auc_val = auc(fpr, tpr)
            aucs.append(auc_val)

            if auc_val > max_auc:
                max_auc = auc_val
                max_acc = acc
                max_f1 = f1
                best_thres = thres
        
        perc_results = {'auc': max_auc, 'acc': max_acc, 'f1': max_f1, 'best_thres': best_thres}
        
        results[perc] = perc_results

    
    return results

#Compute potential Lambda_S values and find their respective F1, accuracy, and AUC score
#function is separated from above as PixelSnail and Transformer implementations take in/produce of differing format
def get_sample_valid_ps(val_anom_loader, val_norm_loader, vqs, params, thres_interval):
    
    raw_scores = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(val_anom_loader)):
            nll = torch.flatten(vqs.loss_sample(x.to(params['device']))).tolist()
            for i in nll:
                raw_scores.append(i)

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(val_norm_loader)):
            nll = torch.flatten(vqs.loss_sample(x.to(params['device']))).tolist()
            for i in nll:
                raw_scores.append(i)

    all_scores = raw_scores.copy()
    all_scores = np.array(all_scores)
    print('Percentiles: ')
    print('20: ' + str(np.percentile(all_scores, 20)))
    print('30: ' + str(np.percentile(all_scores, 30)))
    print('40: ' + str(np.percentile(all_scores, 40)))
    print('50: ' + str(np.percentile(all_scores, 50)))
    print('60: ' + str(np.percentile(all_scores, 60)))
    print('70: ' + str(np.percentile(all_scores, 70)))
    print('80: ' + str(np.percentile(all_scores, 80)))
    print('90: ' + str(np.percentile(all_scores, 90)))
    print('98: ' + str(np.percentile(all_scores, 98)))
    
    percs = []
    percs.append(np.percentile(all_scores, 20))
    percs.append(np.percentile(all_scores, 30))
    percs.append(np.percentile(all_scores, 40))
    percs.append(np.percentile(all_scores, 50))
    percs.append(np.percentile(all_scores, 60))
    percs.append(np.percentile(all_scores, 70))
    percs.append(np.percentile(all_scores, 80))
    percs.append(np.percentile(all_scores, 90))
    percs.append(np.percentile(all_scores, 98))
    
    results = {}
    
    for perc in percs:
        print(perc)
        anom_scores = []
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tqdm(val_anom_loader)):
                nll = torch.flatten(vqs.loss_sample(x.to(params['device']))).tolist()
                score = 0
                for i in nll:
                    raw_scores.append(i)
                    if i > perc:
                        score = score + i
                anom_scores.append(score)

        norm_scores = []
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tqdm(val_norm_loader)):
                nll = torch.flatten(vqs.loss_sample(x.to(params['device']))).tolist()
                score = 0
                for i in nll:
                    raw_scores.append(i)
                    if i > perc:
                        score = score + i
                norm_scores.append(score)
        
        norm = norm_scores
        anom = anom_scores

        df_list = norm+anom
        df = np.array(df_list)

        norm_l = np.array([0] * len(norm_scores)).flatten()
        anom_l = np.array([1] * len(anom_scores)).flatten()
        labels = np.concatenate((norm_l, anom_l), axis=0)

        max_range = max(anom_scores)
        min_range = min(norm_scores)
        thres_range = np.arange (min_range, max_range, thres_interval)
        f1s = []
        accs = []
        aucs = []

        best_thres = 0
        max_f1 = 0
        max_acc = 0
        max_auc = 0

        for thres in tqdm(thres_range):
            preds = []
            for i in df_list:
                if i > thres:
                    preds.append(1)
                else:
                    preds.append(0)
            preds = np.array(preds)

            f1 = f1_score(labels, preds, average='macro')
            f1s.append(f1)
            acc = accuracy_score(labels, preds)
            accs.append(acc)
            fpr, tpr, thresholds = roc_curve(labels, preds)
            auc_val = auc(fpr, tpr)
            aucs.append(auc_val)

            if auc_val > max_auc:
                max_auc = auc_val
                max_acc = acc
                max_f1 = f1
                best_thres = thres
        
        perc_results = {'auc': max_auc, 'acc': max_acc, 'f1': max_f1, 'best_thres': best_thres}
        
        results[perc] = perc_results

    
    return results

#Sample-wise detection inference for models using PixelSnail as AR
def sample_wise_ps(vqs, anom_test_loader, norm_test_loader, params, score_thres):
    anom_scores = []
    raw_scores = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(anom_test_loader)):
            nll = torch.flatten(vqs.loss_sample(x.to(params['device']))).tolist()

            score = 0
            for i in nll:
                raw_scores.append(i)
                if i > score_thres:
                    score = score + i
            anom_scores.append(score)
    print('avg: ' + str(sum(anom_scores)/len(anom_scores)))
    print('max: ' + str(max(anom_scores)))
    print('min: ' + str(min(anom_scores)))

    norm_scores = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(norm_test_loader)):
            nll = torch.flatten(vqs.loss_sample(x.to(params['device']))).tolist()
            score = 0
            for i in nll:
                raw_scores.append(i)
                if i > score_thres:
                    score = score + i
            norm_scores.append(score)

    print('avg: ' + str(sum(norm_scores)/len(norm_scores)))
    print('max: ' + str(max(norm_scores)))
    print('min: ' + str(min(norm_scores)))

    return anom_scores, norm_scores, raw_scores

#Sample-wise detection inference for models using Transformers as AR
def get_sample_scores(anom_test_loader, norm_test_loader, t_model, params, score_threshold):
    print('anomaly sample scores:')
    anom_scores = []
    raw_scores = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(anom_test_loader)):
            x = x.to(params['device'])
            logits, targets = t_model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction = 'none').tolist()
            score = 0
            for i in nll:
                raw_scores.append(i)
                if i > score_threshold:
                    score = score + i
            anom_scores.append(score)

    print('avg: ' + str(sum(anom_scores)/len(anom_scores)))
    print('max: ' + str(max(anom_scores)))
    print('min: ' + str(min(anom_scores)))

    print('normal sample scores:')
    norm_scores = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(norm_test_loader)):
            x = x.to(params['device'])
            logits, targets = t_model(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction = 'none').tolist()
            score = 0
            for i in nll:
                raw_scores.append(i)
                if i > score_threshold:
                    score = score + i
            norm_scores.append(score)

    print('avg: ' + str(sum(norm_scores)/len(norm_scores)))
    print('max: ' + str(max(norm_scores)))
    print('min: ' + str(min(norm_scores)))
    
    return anom_scores, norm_scores, raw_scores

#Evaluate anomaly detection F1 and AUC given a set of anomaly scores for normal and anomalous samples
def evaluate(norm_scores, anom_scores, thres_interval):

    norm = norm_scores
    anom = anom_scores

    df_list = norm+anom
    df = np.array(df_list)

    norm_l = np.array([0] * len(norm_scores)).flatten()
    anom_l = np.array([1] * len(anom_scores)).flatten()
    labels = np.concatenate((norm_l, anom_l), axis=0)

    max_range = max(anom_scores)
    min_range = min(norm_scores)
    thres_range = np.arange (min_range, max_range, thres_interval)
    f1s = []
    accs = []
    aucs = []

    best_thres = 0
    max_f1 = 0
    max_acc = 0
    max_auc = 0

    for thres in tqdm(thres_range):
        preds = []
        for i in df_list:
            if i > thres:
                preds.append(1)
            else:
                preds.append(0)
        preds = np.array(preds)

        f1 = f1_score(labels, preds, average='macro')
        f1s.append(f1)
        acc = accuracy_score(labels, preds)
        accs.append(acc)
        fpr, tpr, thresholds = roc_curve(labels, preds)
        auc_val = auc(fpr, tpr)
        aucs.append(auc_val)

        if auc_val > max_auc:
    #     if f1 > max_f1:
            max_auc = auc_val
            max_acc = acc
            max_f1 = f1
            best_thres = thres

    print('best thres: ' + str(best_thres))
    print('best f1: ' + str(max_f1))
    print('best acc: ' + str(max_acc))
    print('best auc: ' + str(max_auc))

    plt.plot(thres_range, f1s, label='f1')
    plt.plot(thres_range, accs, label='accuracy')
    plt.plot(thres_range, aucs, label='auc')
    plt.legend(loc="upper right")
    plt.title('f1, accuracy, auc values with different thresholds')
    plt.show()
    return best_thres, max_f1, max_acc, max_auc

#For loading images indivisually from dataloader
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#Simple normalise function
def normalise(x):
    return (x-x.min())/(x.max()-x.min())

#Get reconstruction scores from dataloader
def get_rec_scores(model, params, dataloader):
    rec_scores = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(dataloader)):
            x = x.to(params['device'])
            x_hat, _, q_loss = model(x)
            rec_loss = torch.abs(x - x_hat)
            rec_loss = torch.mean(rec_loss).tolist()
            rec_scores.append(rec_loss)

    print('avg: ' + str(sum(rec_scores)/len(rec_scores)))
    print('max: ' + str(max(rec_scores)))
    print('min: ' + str(min(rec_scores)))
    
    return rec_scores

#Sample images and their reconstruction from a dataset via VQ model
def sample_images(params, data_loader, model, count):
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(data_loader):
            if batch_idx == count:
                break
            x = x.to(device=params['device'])
            x_hat, indices, loss = model(x)
            draw_sample_image(x[:6], "Ground-truth images")
            draw_sample_image(x_hat[:6], "Reconstructed images")

#Visualize images side-byside    
def draw_sample_image(x, postfix):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.show()
    
#Load dataloaders with given batch size
def get_datasets(batch_size=1):

    train_direc = '/home2/mvdk66/Anomaly_main/UCSD_Train'
    train_dataset = torchvision.datasets.ImageFolder(root=train_direc ,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),                          
                                    torchvision.transforms.Resize((128,128)),
                                    torchvision.transforms.Normalize([0,0,0], [1,1,1])
                                   ]))
    
    test_anom_direc = '/home2/mvdk66/Anomaly_main/UCSD_Test_anom'
    test_anom_dataset = torchvision.datasets.ImageFolder(root=test_anom_direc ,
                           transform=torchvision.transforms.Compose(
                               [torchvision.transforms.ToTensor(),                          
                                torchvision.transforms.Resize((128,128)),
                                torchvision.transforms.Normalize([0,0,0], [1,1,1])
                               ]))
    
    test_norm_direc = '/home2/mvdk66/Anomaly_main/UCSD_Test_norm'
    test_norm_dataset = torchvision.datasets.ImageFolder(root=test_norm_direc ,
                           transform=torchvision.transforms.Compose(
                               [torchvision.transforms.ToTensor(),                          
                                torchvision.transforms.Resize((128,128)),
                                torchvision.transforms.Normalize([0,0,0], [1,1,1])
                               ]))
    
    val_anom_direc = '/home2/mvdk66/Anomaly_main/UCSD_Test_val_anom'
    val_anom_dataset = torchvision.datasets.ImageFolder(root=val_anom_direc ,
                           transform=torchvision.transforms.Compose(
                               [torchvision.transforms.ToTensor(),                          
                                torchvision.transforms.Resize((128,128)),
                                torchvision.transforms.Normalize([0,0,0], [1,1,1])
                               ]))
    
    val_norm_direc = '/home2/mvdk66/Anomaly_main/UCSD_Test_val_norm'
    val_norm_dataset = torchvision.datasets.ImageFolder(root=val_norm_direc ,
                           transform=torchvision.transforms.Compose(
                               [torchvision.transforms.ToTensor(),                          
                                torchvision.transforms.Resize((128,128)),
                                torchvision.transforms.Normalize([0,0,0], [1,1,1])
                               ]))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_anom_loader = torch.utils.data.DataLoader(test_anom_dataset, batch_size=batch_size)
    test_norm_loader = torch.utils.data.DataLoader(test_norm_dataset, batch_size=batch_size)
    
    val_anom_loader = torch.utils.data.DataLoader(val_anom_dataset, batch_size=batch_size)
    val_norm_loader = torch.utils.data.DataLoader(val_norm_dataset, batch_size=batch_size)
    
    return train_loader, test_anom_loader, test_norm_loader, val_anom_loader, val_norm_loader

