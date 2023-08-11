from __future__ import print_function, division
import os
import argparse
import time
import pickle
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.optim as optim
from LSD_EBM_code.model_Unet_vae import ReconNet, VAE_new
from LSD_EBM_code.dataset_vae import CSI_Dataset
import nibabel as nib
import torch.nn as nn
from medpy.metric.binary import dc


modes=['train','validation']

def seg_loss(pred, target, weight, weightFP):
    
    FP = torch.sum(weightFP*(1-target)*pred)
    FN = torch.sum(weight*(1-pred)*target)

    return FP, FN

def VaeLoss(recon_x,x,mu,logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')#/10000000
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# * 100.
    print("BCE: %.4f\tKLD: %.4f\t" % (BCE.item()/batch_size, KLD.item()/batch_size))
    loss = BCE+KLD
    return loss

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


# Training args
parser = argparse.ArgumentParser(description='Fully Convolutional Network')

parser.add_argument('--num_epochs',                         default=201, help='Number of epochs')                      
parser.add_argument('--vae_lr', type=float,                 default=0.00002,help='learning rate (default: 0.001)')
parser.add_argument('--save_model', action='store_true',    default=True,help='For Saving the current Model')
parser.add_argument('--train_set',                          default='XXX_bodies_data_train',help='name of dataset path')
parser.add_argument('--validation_set',                     default='XXX_bodies_data_validation',help='name of validation-set path')
parser.add_argument('--test_set',                           default='XXX_bodies_data_test',help='name of testset path')
parser.add_argument('--experiment',                         default='Test',help='name of experiment')
parser.add_argument('--load_from_ep',                       default=200, type=int, help='checkpoint you want to load for the models')
parser.add_argument('--epoch_start',                        default=0, type=int, help='epoch you want to start from')
parser.add_argument('--save_visu',                          default=True, help='saves on training/testing visualization')
parser.add_argument('--num_channels',                       default=16, type=int,help='Number of Channels for the CNN')
parser.add_argument('--num_latents',                        default=100, type=int,help='dimension of latent space')
parser.add_argument('--batch_size',                         default=4, type=int,help='Batch Size')
# Testing
parser.add_argument('--test_perf',                          default=True, help='if prediction should be performed on test dataset')
args = parser.parse_args()

run_name = "VAE_Vert_01"
main_path ="./../Vert_data/" + run_name + "/" 

if not os.path.exists(main_path):
    os.makedirs(main_path)

with open(main_path + "hyper_para.txt", "w") as output:  ## creates new file but empty
    for arg in vars(args):
        print(arg, getattr(args, arg))
        output.write(str(arg) + "\t" + str(getattr(args, arg)) + "\n")

out_f = open(main_path + "output.txt",'w')

print("Pytorch Version:", torch.__version__, file=out_f)
print("Experiment: "+args.experiment, file=out_f)
print(args, file=out_f)

# Use GPU if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size


class VAE_model(object):
    def __init__(self):
        # Initialize networks
        self.vaeNet=VAE_new(args.num_channels,args.num_latents).to(device) # Since the images are b&w, they initially have 1 ch
    
        # use multiple gpus
        n_gpus=torch.cuda.device_count()
        if n_gpus>1:
            print("Let's use", n_gpus, "GPUs!", file=out_f)
            batch_size = n_gpus*args.batch_size
            disNet=nn.DataParallel(VAE_new(args.num_channels,args.num_latents).to(device))
    
    def load_model(self, base_model_path):
        self.vaeNet.load_state_dict(torch.load(os.path.join(base_model_path, 'experiments', 'vae_' + str(args.load_from_ep)+'.pth')))
        print("LOAD VAE MODEL EP ")

        p = 0
        for pms in self.vaeNet.parameters():
            p += torch.numel(pms)
        print("vaeNet num parameters: ", p, file=out_f)

        self.vaeNet.eval()

    def reconstruction(self, x):
        mean,logvar,vae_recon_patch = self.vaeNet(x)
        pred_patch = torch.round(vae_recon_patch).detach() 
        return pred_patch
    
