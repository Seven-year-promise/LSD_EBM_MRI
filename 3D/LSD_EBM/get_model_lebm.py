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
from LSD_EBM_code.model_Unet_lebm import ReconNet, E_func, GenModel_vert, sample_langevin_prior_z, Buff
from LSD_EBM_code.dataset_vae import CSI_Dataset
import nibabel as nib
import torch.nn as nn
from medpy.metric.binary import dc

modes=['train', 'validation']


mse = nn.MSELoss(reduction='sum')

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

def sample_langevin_post_z(z0, x, netG, netE, g_l_steps=20, g_l_step_size=0.1, e_prior_sig=1.,  noise_sig=1., g_llhd_sigma=0.3, gamma=1., g_l_with_noise=True, verbose=False, hidden=False):
    netE.eval()
    netG.eval()
    for p in netE.parameters():
        p.requires_grad = False
    for p in netG.parameters():
        p.requires_grad = False
    z = z0.clone().detach()
    z.requires_grad = True
    
    noise = torch.randn(z.shape, device=z.device)

    zs = []
    zs.append(z0.clone().detach())
    
    for i in range(g_l_steps):        # mcmc iteration

        x_hat = netG.forward(z)               # x = gen_nw(z)
        # log lkhd (x|z) = 1/2 * (x-x_true) / sigma^2
        log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * mse(x_hat, x)
        grad_log_lkhd = torch.autograd.grad(log_lkhd, z)[0]
        
        en = netE.forward(z)                     # E(z)
        grad_log_prior_pt1 = torch.autograd.grad(en.mean(), z)[0]
        grad_log_prior_pt2 = 1.0 / (e_prior_sig * e_prior_sig) * z.data.detach()

        # gradLogP(z|x) = gradLogP(x|z) + gradLogP(z) - gradLogP(x) with last term =0
        z.data = z.data - 0.5 * g_l_step_size * (grad_log_lkhd + grad_log_prior_pt1 + grad_log_prior_pt2)
        # z.data.add_(- 0.5 * g_l_step_size * (z_grad_g + z_grad_e + 1.0 / (e_prior_sig * e_prior_sig) * z.data))
        # .. + s * N(0,1).sample()
        if g_l_with_noise:
            noise.normal_(0, noise_sig)
            z.data.add_(np.sqrt(g_l_step_size) * torch.randn_like(z).data)

        g_l_step_size *= gamma

        zs.append(z.clone().detach())

    if hidden:
        return zs
    else:
        return z.detach() 

fixed_images = ['L1_verse033_seg.nii.gz', 'L1_verse265_seg.nii.gz', 'L2_verse145_seg.nii.gz',  'L3_verse082_seg.nii.gz']
    
# Training args
parser = argparse.ArgumentParser(description='Fully Convolutional Network')

# Trianing
parser.add_argument('--num_epochs',                         default=0, type=int, help='Number of epochs')                      
parser.add_argument('--batch_size',                         default=2, type=int,help='Batch Size')
parser.add_argument('--ebm_lr', type=float,                 default=0.0001,help='learning rate of the EMB network')
parser.add_argument('--gen_lr', type=float,                 default=0.0001,help='learning rate of the generation network')
parser.add_argument('--load_from_ep',                       default=200, type=int, help='checkpoint you want to load for the models')
parser.add_argument('--epoch_start',                        default=200, type=int, help='epoch you want to start from')
parser.add_argument('--recon_loss',                         default='bce', help='reconstruction loss of choice (mse, bce)')
parser.add_argument('--ebm_dyn_lr',                         default=None, type=float, help='if learning rate of ebm model should be set dynamically')
parser.add_argument('--gen_dyn_lr',                         default=None, type=float, help='if learning rate of generation model should be set dynamically')
parser.add_argument('--EBM_reg',                            default=0, type=float, help='regularization applied to the latent EBM loss')
# Data
parser.add_argument('--save_model', action='store_true',    default=True,help='For Saving the current Model')
parser.add_argument('--train_set',                          default='XXX_bodies_data_train',help='name of dataset path')
parser.add_argument('--validation_set',                     default='XXX_bodies_data_validation',help='name of validation-dataset path')
parser.add_argument('--test_set',                           default='XXX_bodies_data_test',help='name of testset path')
parser.add_argument('--experiment',                         default='Test',help='name of experiment')
parser.add_argument('--save_visu',                          default=True,help='saves on training/testing visualization')
parser.add_argument('--ep_betw_val',                        default=1, type=int,help='Number of training epochs between two validation steps')
# Models
parser.add_argument('--num_channels',                       default=16, type=int,help='Number of Channels for the CNN')
parser.add_argument('--num_latents',                        default=100, type=int,help='dimension of latent space')
parser.add_argument('--prior_var',                          default=1., type=float,help='Variance of the prior distribution of the EBM')
parser.add_argument('--prior_bn',                           default=True, help='If batch normalization should be applied in the prior model')
parser.add_argument('--gen_var',                            default=0.3, type=float,help='Assumed variance of the generation model')
parser.add_argument('--ebm_num_layers',                     default=3, type=int,help='Number of layers of the EBM prior network')
parser.add_argument('--ebm_inner_dim',                      default=300, type=int,help='Number of neurons of inner layers of the EBM prior network')
# Sampling
parser.add_argument('--use_samp_buff',                      default=False, help='if a sample buffer should be used for the MCMC sampling')
parser.add_argument('--samp_buff_size',                     default=256, type=int,help='Max number of elements in the sample buffer')
parser.add_argument('--pr_mcmc_steps_tr',                   default=10, type=int,help='Number of mcmc steps sampling from prior for training') #60
parser.add_argument('--pr_mcmc_steps_val',                  default=10, type=int,help='Number of mcmc steps sampling from prior for validation') #60
parser.add_argument('--pr_mcmc_step_size',                  default=0.4, type=float,help='Step size of mcmc steps sampling from prior')
parser.add_argument('--pr_mcmc_noise_var',                  default=1., type=float,help='Variance of noise added to mcmc steps sampling from prior')
parser.add_argument('--po_mcmc_steps_tr',                   default=20, type=int,help='Number of mcmc steps sampling from posterior for training')
parser.add_argument('--po_mcmc_steps_val',                  default=20, type=int,help='Number of mcmc steps sampling from posterior for validation')
parser.add_argument('--po_mcmc_steps_test',                 default=20,type=int,help='Number of mcmc steps sampling from posterior for testing')
parser.add_argument('--po_mcmc_step_size',                  default=0.1, type=float,help='Step size of mcmc steps sampling from posterior')
parser.add_argument('--po_mcmc_noise_var',                  default=1., type=float,help='Variance of noise added to mcmc steps sampling from posterior')
parser.add_argument('--pr_mcmc_step_gamma',                 default=1., type=float,help='Factor to decrease prior sample step size per step')
parser.add_argument('--po_mcmc_step_gamma',                 default=1., type=float,help='Factor to decrease posterior sample step size per step')
# Testing
parser.add_argument('--test_perf',                          default=True, help='if prediction should be performed on test dataset')

args = parser.parse_args()

def sample_p_0(n, sig=args.prior_var):
    return sig * torch.randn(*[n, args.num_latents])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%Main
class LEBM_model(object): 
    def __init__(self):  
    
        # Use GPU if it is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = args.batch_size
        
        # Initialize networks and buffer
        e_func    = E_func(args.num_latents, args.ebm_inner_dim, args.ebm_num_layers, batchnorm=args.prior_bn).to(device)
        gen_model = GenModel_vert(args.num_channels, args.num_latents).to(device)


        buff = None
        if args.use_samp_buff:
            buff = Buff(sample_p_0(args.batch_size), args.samp_buff_size)
            buff.add_samp(sample_p_0(5))
        
        # use multiple gpus
        n_gpus=torch.cuda.device_count()

        self.e_func = e_func
        self.gen_model = gen_model

    def load_model(self, base_model_path = ""):
        e_func_path = os.path.join(base_model_path,'experiments', 'efunc_'+str(args.load_from_ep)+'.pth')
        if os.path.exists(e_func_path):
            self.e_func.load_state_dict(torch.load(e_func_path))
            print("LOAD EBM MODEL EP ", args.load_from_ep)
            self.gen_model.load_state_dict(torch.load(os.path.join(base_model_path,'experiments', 'genmodel_'+str(args.load_from_ep)+'.pth')))
            print("LOAD GENERATION MODEL", args.load_from_ep)
        else:
            e_func_path = os.path.join(base_model_path,'experiments', 'efunc_180.pth')
            self.e_func.load_state_dict(torch.load(e_func_path))
            print("LOAD EBM MODEL EP ", 180)
            self.gen_model.load_state_dict(torch.load(os.path.join(base_model_path,'experiments', 'genmodel_180.pth')))
            print("LOAD GENERATION MODEL", 180)

        self.e_func.eval()
        self.gen_model.eval()

    def generation(self, noise, step):
        z_e = sample_langevin_prior_z(noise, self.e_func, step, args.pr_mcmc_step_size, args.prior_var, args.pr_mcmc_noise_var, args.pr_mcmc_step_gamma)
        recon_patch = self.gen_model.forward(z_e)
        pred_patch = torch.round(recon_patch).detach()
        return pred_patch

    def reconstruction(self, x):
        z_g_0 = sample_p_0(1).to(device)  
        
        z_g = sample_langevin_post_z(z_g_0, x, self.gen_model, self.e_func, args.po_mcmc_steps_test, args.po_mcmc_step_size, args.prior_var, args.po_mcmc_noise_var, args.gen_var, args.po_mcmc_step_gamma)
        recon_patch = self.gen_model.forward(z_g)
        pred_patch = torch.round(recon_patch).detach() 

        return pred_patch

    def get_all_latents(self, x):
        z_g_0 = sample_p_0(1).to(device)  
        zs = sample_langevin_post_z(z_g_0, x, self.gen_model, self.e_func, args.po_mcmc_steps_test, args.po_mcmc_step_size, args.prior_var, args.po_mcmc_noise_var, args.gen_var, args.po_mcmc_step_gamma, hidden=True)
        
        return zs

    def z_latent(self, noise, step):
        z_e = sample_langevin_prior_z(noise, self.e_func, step, args.pr_mcmc_step_size, args.prior_var, args.pr_mcmc_noise_var, args.pr_mcmc_step_gamma)
        
        return z_e  
   