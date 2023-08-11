from __future__ import print_function, division
import os
import argparse
import time
import pickle
from torch.utils.data import DataLoader
import numpy as np
import scipy

import torch
import torch.optim as optim
from LSD_EBM_code.model_Unet_lsdebm import E_func, VAE_new_Enc_small, VAE_new_Dec_small
from LSD_EBM_code.dataset_vae import CSI_Dataset
import nibabel as nib
import torch.nn as nn
from medpy.metric.binary import dc

modes=['train', 'validation']

## added by yanke
def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def DiceCoeff(pred, gt):
     pred = pred.to('cpu').numpy()
     gt = gt.to('cpu').numpy()
     
     #if gt is all zero (use inverse to count)
     if np.count_nonzero(gt) == 0:
         gt = gt+1
         pred = 1-pred
         
     dice_score = dc(pred.astype(np.int),gt.astype(np.int)) 
     return dice_score
 
def DiceCoeffProc(pred, gt, body):
    pred = pred.to('cpu').numpy()
    gt = gt.to('cpu').numpy()
    body = body.to('cpu').numpy()
    
    pred = pred-body
    gt = gt-body
 
    #if gt is all zero (use inverse to count)
    if np.count_nonzero(gt) == 0:
        gt = gt+1
        pred = 1-pred
     
    dice_score = dc(pred.astype(np.int),gt.astype(np.int)) 
    return dice_score


def load_fixed(img_names, dataset_path):
    full=[]
    body=[]
    for img_name in img_names:
        body_name = img_name.split('.')[0]+'_weight.nii.gz'

        img_file = os.path.join(dataset_path,  'full',  img_name)
        full_patch = np.round(nib.load(img_file).get_fdata()).astype('uint8')
        
        body_file = os.path.join(dataset_path, 'border',  body_name)
        body_patch = np.round(nib.load(body_file).get_fdata()).astype('uint8')
        
        full_patch = np.expand_dims(full_patch.astype('uint8'), axis=(0,1))
        body_patch = np.expand_dims(body_patch.astype('uint8'), axis=(0,1))
        
        full.append(torch.from_numpy(full_patch))
        body.append(torch.from_numpy(body_patch))
    
    return full, body

def extract(a, t, shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    if isinstance(t, int) or len(t.shape) == 0:
      t = torch.ones(shape[0]) * t
    bs = t.size()[0]
    assert shape[0] == bs
    out = torch.gather(a, 0 ,t)
    assert out.shape[0] == bs
    out = torch.stack([out]*shape[1], axis=1)
    return out
    # return out.view(bs, ((len(shape) - 1) * [1]))

def sample_langevin_cond_z(z_tilde, t, netE, e_l_steps, e_l_step_size, e_prior_sig=1., noise_sig=0.01, e_l_with_noise = True, verbose=False):
        sigma = extract(torch.tensor(sigmas).to(z_tilde.device), t + 1, z_tilde.size())
        sigma_cum = extract(torch.tensor(sigmas_cum).to(z_tilde.device), t, z_tilde.size())
        a_s = extract(torch.tensor(a_s_prev).to(z_tilde.device), t + 1, z_tilde.size())

        netE.eval()
        for p in netE.parameters():
            p.requires_grad = False
        y = torch.randn(z_tilde.size()).to(z_tilde.device)
        y.requires_grad = True

        noise = torch.randn(y.size()).to(z_tilde.device)
        for i in range(e_l_steps):                        # mcmc iteration
            en_y = netE.forward(y, t)
            # print("*e ", en_y)
            # -logp = ...
            logp = en_y.sum() + ((y - z_tilde) ** 2 / 2 / sigma ** 2).sum() 
            logp.backward()

            y.data.add_(- 0.5 * e_l_step_size * y.grad.data) 

            if e_l_with_noise:
                noise.normal_(0, noise_sig)
                y.data.add_(np.sqrt(e_l_step_size) * noise.data)

            y.grad.detach_()
            y.grad.zero_()

        for p in netE.parameters():
            p.requires_grad = True
        netE.train()
        z = (y / a_s).float()
        return z.detach()

def q_sample(z_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    if noise is None:
      noise = torch.randn(z_start.size()).to(t.device)
    assert noise.shape == z_start.shape
    z_t = extract(torch.tensor(a_s_cum).to(t.device), t, z_start.size()) * z_start + \
          extract(torch.tensor(sigmas_cum).to(t.device), t, z_start.size()) * noise *0.1 #CHANGED

    return z_t

"""
For latent vecotrs (b x latent_dim) and global T=#diffusion timesteps creates
T x b x latent_dim vetors with increased noise. (FORWARD PROCESS)
"""
def q_sample_progressive(z_0, diff_timesteps):
    """
    Generate a full sequence of disturbed latent vectors
    """
    z_preds = []
    for t in range(diff_timesteps + 1):
      t_now = (torch.ones(z_0.size()[0], dtype=int) * t).to(z_0.device)
      z = q_sample(z_0, t_now)
      z_preds.append(z)
    z_preds = torch.stack(z_preds, axis=0)

    return z_preds

# Starting from noise latent vector (b x latent_dim) 
#   returns (diff_steps x b x latent_dim) with noise vectors last and reduced noise via MCMC sampling towards beginning
def p_sample_progressive(noise, e_func, latent_dim, hor, e_l_steps, e_l_step_size, hidden=False):
    """
    Sample a sequence of latent vectors with the sequence of noise levels
    """
    zs = []
    num = noise.shape[0]
    z_neg_t = noise # b x latent_dim
    z_neg = torch.zeros((hor,) + noise.size()).to(noise.device) # diff_timesteps x b x latent_dim
    z_neg = torch.cat([z_neg, noise.view(1, num, -1)], axis=0) # (diff_timesteps+1) x b x latent_dim # TODO: here changed
    for t in range(hor - 1, -1, -1):
      # print("pre ", e_func.forward(z_neg_t, torch.tensor([t]*num).to(noise.device)).detach().numpy().flatten())
      z_neg_t = sample_langevin_cond_z(z_neg_t, torch.tensor([t]*num).to(noise.device), e_func, e_l_steps, e_l_step_size) # b x latent_dim
      z_neg_t = z_neg_t.view(num, latent_dim) # useless?
      # print("post ", e_func.forward(z_neg_t, torch.tensor([t]*num).to(noise.device)).detach().numpy().flatten())
      insert_mask = (torch.ones(hor+1)*t == torch.range(0, hor)).float().to(noise.device)
      insert_mask = torch.stack([torch.stack([insert_mask]*noise.size()[1], axis=1)]*noise.size()[0], axis=1) # latent_timesteps x b x latent_dim
      # print(torch.stack([z_neg_t]*(diff_timesteps+1)).size())
      z_neg = insert_mask * torch.stack([z_neg_t]*(hor+1)) + (1. - insert_mask) * z_neg
      zs.append(z_neg_t)
    if hidden:
        return zs
    else:
        return z_neg

# Create diffused samples at t and t+1 diffusion steps
# (noise at t+1 > noise at t)
def q_sample_pairs(z_start, t):
    """
    Generate a pair of disturbed latent vectors for training
    :param z_start: x_0
    :param t: time step t
    :return: z_t, z_{t+1}
    """
    noise = torch.randn(z_start.size()).to(z_start.device)
    z_t = q_sample(z_start, t)
    z_t_plus_one = extract(torch.tensor(a_s).to(z_start.device), t+1, z_start.size()) * z_t + \
                   extract(torch.tensor(sigmas).to(z_start.device), t+1, z_start.size()) * noise*0.1

    return z_t, z_t_plus_one

 
    
fixed_images = ['L1_verse033_seg.nii.gz', 'L1_verse265_seg.nii.gz', 'L2_verse145_seg.nii.gz',  'L3_verse082_seg.nii.gz']

# Training args
parser = argparse.ArgumentParser(description='Fully Convolutional Network')

# Trianing
parser.add_argument('--num_epochs',                         default=0, type=int, help='Number of epochs')                      
parser.add_argument('--batch_size',                         default=4, type=int,help='Batch Size')
parser.add_argument('--ebm_lr', type=float,                 default=0.00002,help='learning rate of the EMB network')
#parser.add_argument('--ebm_lr', type=float,                 default=0.0001,help='learning rate of the EMB network') # like vert lebm, like classis ds
parser.add_argument('--enc_lr', type=float,                 default=0.00002,help='learning rate of the encoder network')
parser.add_argument('--dec_lr', type=float,                 default=0.00002,help='learning rate of the decoder network')
parser.add_argument('--load_from_ep',                       default=200, type=int, help='checkpoint you want to load for the models') # 200
parser.add_argument('--epoch_start',                        default=200, type=int, help='epoch you want to start from')
parser.add_argument('--recon_loss',                         default='bce', help='reconstruction loss of choice (mse, bce)')
parser.add_argument('--ebm_dyn_lr',                         default=None, type=float, help='if learning rate of ebm model should be set dynamically')
parser.add_argument('--enc_dyn_lr',                         default=None, type=float, help='if learning rate of ebm model should be set dynamically')
parser.add_argument('--dec_dyn_lr',                         default=None, type=float, help='if learning rate of generation model should be set dynamically')
parser.add_argument('--EBM_reg',                            default=0, type=float, help='regularization applied to the latent EBM loss')
# Data
parser.add_argument('--save_model', action='store_true',    default=True,help='For Saving the current Model')
parser.add_argument('--train_set',                          default='XXX_bodies_data_train',help='name of dataset path')
parser.add_argument('--validation_set',                     default='XXX_bodies_data_validation',help='name of validation-dataset path')
parser.add_argument('--test_set',                           default='XXX_bodies_data_test',help='name of testset path')
parser.add_argument('--experiment',                         default='Test',help='name of experiment')
parser.add_argument('--save_visu',                          default=True,help='saves on training/testing visualization')
parser.add_argument('--ep_betw_val',                        default=2, type=int,help='Number of training epochs between two validation steps')
# Models
parser.add_argument('--num_channels',                       default=16, type=int,help='Number of Channels for the CNN')
parser.add_argument('--num_latents',                        default=100, type=int,help='dimension of latent space')
parser.add_argument('--prior_var',                          default=1., type=float,help='Variance of the prior distribution of the EBM')
parser.add_argument('--gen_var',                            default=0.3, type=float,help='Assumed variance of the generation model')
parser.add_argument('--ebm_num_layers',                     default=1, type=int,help='Number of layers of the EBM prior network')
parser.add_argument('--ebm_inner_dim',                      default=1000, type=int,help='Number of neurons of inner layers of the EBM prior network')
parser.add_argument('--prior_bn',                           default=False, help='If batch normalization should be applied in the prior model')
# Sampling
parser.add_argument('--cond_mcmc_steps_tr',                 default=300,  type=int,help='Number of mcmc steps sampling from prior for training')
parser.add_argument('--cond_mcmc_steps_val',                default=300,  type=int,help='Number of mcmc steps sampling from prior for validation')
parser.add_argument('--cond_mcmc_step_size',                default=0.01, type=float,help='Step size of mcmc steps sampling from prior')
# Diffusion
parser.add_argument('--diff_timesteps',                     default=30, type=int, help='Number of diffusion timesteps') #30
parser.add_argument('--beta_start',                         default=0.0001, type=float,help='Diffusion schedule')
parser.add_argument('--beta_end',                           default=0.01, type=float,help='Diffusion schedule')

# Testing
parser.add_argument('--test_perf',                          default=True, help='if prediction should be performed on test dataset')

args = parser.parse_args()
   

# Use GPU if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size

# diffusion 
betas = np.linspace(args.beta_start, args.beta_end, 1000)
betas = np.append(betas, 1.)
sqrt_alphas = np.sqrt(1. - betas)
idx = np.concatenate([np.arange(args.diff_timesteps) * (1000 // ((args.diff_timesteps - 1) * 2)), [999]])
a_s = np.concatenate(
    [[np.prod(sqrt_alphas[: idx[0] + 1])],
    np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
sigmas = np.sqrt(1 - a_s ** 2)
a_s_cum = np.cumprod(a_s)
sigmas_cum = np.sqrt(1 - a_s_cum ** 2)
a_s_prev = a_s.copy()
a_s_prev[-1] = 1


# Initialize networks 
#e_func    = E_func(args.num_latents, args.ebm_inner_dim, args.ebm_num_layers, batchnorm=args.prior_bn).to(device)
#enc_model = VAE_new_Enc_small(args.num_channels,args.num_latents).to(device) 
#dec_model = VAE_new_Dec_small(args.num_channels,args.num_latents).to(device) 


def sample_p_0(n, sig=args.prior_var):
    return sig * torch.randn(*[n, args.num_latents])
    
# use multiple gpus
n_gpus=torch.cuda.device_count()
    
"""
#load models if requested

def load_model(noise, step, base_model_path="./Vert_data/LSD-EBM_Vert_step_30"):

    e_func.load_state_dict(torch.load(os.path.join(base_model_path,'experiments', 'efunc_'+str(args.load_from_ep)+'.pth')))
    print("LOAD EBM MODEL EP ", base_model_path, args.load_from_ep)
    enc_model.load_state_dict(torch.load(os.path.join(base_model_path,'experiments', 'enc_model_'+str(args.load_from_ep)+'.pth')))
    print("LOAD Encoder MODEL", base_model_path, args.load_from_ep)
    dec_model.load_state_dict(torch.load(os.path.join(base_model_path,'experiments', 'dec_model_'+str(args.load_from_ep)+'.pth')))
    print("LOAD Decoder MODEL", base_model_path, args.load_from_ep)

    
    p = 0
    for pms in self.e_func.parameters():
        p += torch.numel(pms)

    p = 0
    for pms in self.enc_model.parameters():
        p += torch.numel(pms)
    p = 0
    for pms in self.dec_model.parameters():
        p += torch.numel(pms)
    
    e_func.eval()
    enc_model.eval()
    dec_model.eval()

    z_gen  = p_sample_progressive(noise, self.e_func, latent_dim=args.num_latents, hor=step, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :].detach()
    with torch.no_grad():
        gen_patch = self.dec_model.forward(z_gen).detach().cpu()
        gen_patch = torch.round(gen_patch)
    return gen_patch

"""

#%%Main
class LSD_EBM_model(object):
    def __init__(self):
        # Initialize networks 
        e_func    = E_func(args.num_latents, args.ebm_inner_dim, args.ebm_num_layers, batchnorm=args.prior_bn).to(device)
        enc_model = VAE_new_Enc_small(args.num_channels,args.num_latents).to(device) 
        dec_model = VAE_new_Dec_small(args.num_channels,args.num_latents).to(device)  

        

        self.e_func = e_func
        self.enc_model = enc_model
        self.dec_model = dec_model

    def load_model(self, base_model_path="./Vert_data/LSD-EBM_Vert_step_30"):

        self.e_func.load_state_dict(torch.load(os.path.join(base_model_path,'experiments', 'efunc_'+str(args.load_from_ep)+'.pth')))
        print("LOAD EBM MODEL EP ", base_model_path, args.load_from_ep)
        self.enc_model.load_state_dict(torch.load(os.path.join(base_model_path,'experiments', 'enc_model_'+str(args.load_from_ep)+'.pth')))
        print("LOAD Encoder MODEL", base_model_path, args.load_from_ep)
        self.dec_model.load_state_dict(torch.load(os.path.join(base_model_path,'experiments', 'dec_model_'+str(args.load_from_ep)+'.pth')))
        print("LOAD Decoder MODEL", base_model_path, args.load_from_ep)

        
        p = 0
        for pms in self.e_func.parameters():
            p += torch.numel(pms)

        p = 0
        for pms in self.enc_model.parameters():
            p += torch.numel(pms)
        p = 0
        for pms in self.dec_model.parameters():
            p += torch.numel(pms)
        
        self.e_func.eval()
        self.enc_model.eval()
        self.dec_model.eval()

    def generation(self, noise, step):
        z_gen  = p_sample_progressive(noise, self.e_func, latent_dim=args.num_latents, hor=step, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :].detach()
        with torch.no_grad():
            gen_patch = self.dec_model.forward(z_gen).detach().cpu()
            gen_patch = torch.round(gen_patch)
        return gen_patch

    def reconstruction(self, x, step):
        z_mu, z_logvar = self.enc_model.forward(x.to(device).float())
        z_inf = z_mu + torch.exp(0.5*z_logvar)*torch.randn_like(z_mu)
    
        # diffusion
        t_diff = step #  int(step/6)*2 set number of diff. steps to perform 
        val_z_diff     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * t_diff).detach()
        val_z_recon    = p_sample_progressive(val_z_diff, self.e_func, latent_dim=args.num_latents, hor=t_diff, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
    
        recon_patch = self.dec_model.forward(val_z_recon.float())
        pred_patch = torch.round(recon_patch).detach() 
        return pred_patch

    def get_all_latents(self, x, step=0):
        z_mu, z_logvar = self.enc_model.forward(x.to(device).float())
        z_inf = z_mu + torch.exp(0.5*z_logvar)*torch.randn_like(z_mu)
        z_diffusion = []
        z_diffusion.append(z_inf)
        
        t_diff = step #  int(step/6)*2 set number of diff. steps to perform 
        for t in range(1, t_diff+1):
            z_t = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * t).detach()
            z_diffusion.append(z_t)
        val_z_diff = z_diffusion[-1]
        
        z_denoising = p_sample_progressive(val_z_diff, self.e_func, latent_dim=args.num_latents, hor=t_diff, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size, hidden=True)
    
        return z_diffusion, z_denoising


    def z_latent(self, noise, step):
        z_gen  = p_sample_progressive(noise, self.e_func, latent_dim=args.num_latents, hor=step, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :].detach()
        
        return z_gen

    def post_z(self, x, step=0):
        z_mu, z_logvar = self.enc_model.forward(x.to(device).float())
        z_inf = z_mu + torch.exp(0.5*z_logvar)*torch.randn_like(z_mu)
    
        # diffusion
        t_diff = step #  int(step/6)*2 set number of diff. steps to perform 
        val_z_diff     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * t_diff).detach()
        val_z_recon    = p_sample_progressive(val_z_diff, self.e_func, latent_dim=args.num_latents, hor=t_diff, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
    
        return val_z_recon.float() 
    
    

