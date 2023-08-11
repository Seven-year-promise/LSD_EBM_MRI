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
from time import process_time
from get_model_lsd_ebm import LSD_EBM_model
from get_model_lsd_ebm import args as lsd_ebm_args

import get_model_lsd_ebm

#%%Main
if  __name__ == "__main__" :   
    run_name = "compare_steps_lsd_ebm"
    main_path="./comp_results/" + run_name + "/" 

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    with open(main_path + "hyper_para.txt", "w") as output:  ## creates new file but empty
        for arg in vars(lsd_ebm_args):
            print(arg, getattr(lsd_ebm_args, arg))
            output.write(str(arg) + "\t" + str(getattr(lsd_ebm_args, arg)) + "\n")

    out_f = open(main_path + "output.txt",'w')

    if not os.path.exists(main_path):
        print('Creating directory at:', main_path, file=out_f)
        os.makedirs(main_path)
    
    print("Pytorch Version:", torch.__version__, file=out_f)
    print("Experiment: "+lsd_ebm_args.experiment, file=out_f)
    print(lsd_ebm_args, file=out_f)
    print("Main path: ", main_path, file=out_f)
    
    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = lsd_ebm_args.batch_size
    
    # diffusion 
    lsd_ebm_model = LSD_EBM_model()
    """

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
    e_func    = E_func(args.num_latents, args.ebm_inner_dim, args.ebm_num_layers, batchnorm=args.prior_bn).to(device)
    enc_model = VAE_new_Enc_small(args.num_channels,args.num_latents).to(device) 
    dec_model = VAE_new_Dec_small(args.num_channels,args.num_latents).to(device) 


    def sample_p_0(n, sig=args.prior_var):
        return sig * torch.randn(*[n, args.num_latents])
        
    # use multiple gpus
    n_gpus=torch.cuda.device_count()

    """
        
    #load models if requested
    steps = [2, 15, 20, 30]
    base_model_path = "./Vert_data/LSD-EBM_Vert_step_"

    z_noise = torch.randn((1, lsd_ebm_args.num_latents)).to(device) # generate the same noise

    for step in steps:
        print("LOAD EBM MODEL EP ", base_model_path+str(step), lsd_ebm_args.load_from_ep, file=out_f)
        lsd_ebm_model.load_model(base_model_path+str(step))
        #load_model()
        """
        #e_func.load_state_dict(torch.load(os.path.join(base_model_path+str(step),'experiments', 'efunc_'+str(args.load_from_ep)+'.pth')))
        #print("LOAD EBM MODEL EP ", base_model_path+str(step), args.load_from_ep, file=out_f)
        #enc_model.load_state_dict(torch.load(os.path.join(base_model_path+str(step),'experiments', 'enc_model_'+str(args.load_from_ep)+'.pth')))
        #print("LOAD Encoder MODEL", base_model_path+str(step), args.load_from_ep, file=out_f)
        #dec_model.load_state_dict(torch.load(os.path.join(base_model_path+str(step),'experiments', 'dec_model_'+str(args.load_from_ep)+'.pth')))
        #print("LOAD Decoder MODEL", base_model_path+str(step), args.load_from_ep, file=out_f)

    
        p = 0
        for pms in e_func.parameters():
            p += torch.numel(pms)
        print("e_func num parameters: ", p, file=out_f)

        p = 0
        for pms in enc_model.parameters():
            p += torch.numel(pms)
        print("enc_model num parameters: ", p, file=out_f)
        p = 0
        for pms in dec_model.parameters():
            p += torch.numel(pms)
        print("dec_model num parameters: ", p, file=out_f)
        """

        t1_start = process_time() 
        z_gen = None
        gen_patch = None

        # Generation
        
        gen_patch = lsd_ebm_model.generation(z_noise, step)

        with torch.no_grad():
            nib.save(nib.Nifti1Image(np.squeeze(gen_patch.to('cpu').detach().numpy()[0]),affine=None), main_path+ "/gen_patch_step_" + str(step) + ".nii.gz")
        #del gen_patch  # free memory on GPU
        t1_stop = process_time() 

        # the geration is bad, as the codes are changed

        print("step ", step, "processing time (s)", t1_stop-t1_start, file=out_f)
                    
    out_f.close()

