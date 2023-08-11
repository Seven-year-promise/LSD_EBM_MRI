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
from get_model_lebm import LEBM_model
from get_model_lebm import args as lebm_args
from get_model_ddpm import DDPM_model
from get_model_ddpm import args as ddpm_args

import get_model_lsd_ebm as lsd_ebm_func


#%%Main
if  __name__ == "__main__" :   
    run_name = "compare_generations_lsdebm_lebm_ddpm"
    main_path="./comp_results/" + run_name + "/" 

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    with open(main_path + "hyper_para_lsd_ebm.txt", "w") as output:  ## creates new file but empty
        for arg in vars(lsd_ebm_args):
            print(arg, getattr(lsd_ebm_args, arg))
            output.write(str(arg) + "\t" + str(getattr(lsd_ebm_args, arg)) + "\n")

    with open(main_path + "hyper_para_lebm.txt", "w") as output:  ## creates new file but empty
        for arg in vars(lebm_args):
            print(arg, getattr(lebm_args, arg))
            output.write(str(arg) + "\t" + str(getattr(lebm_args, arg)) + "\n")
        
    with open(main_path + "hyper_para_ddpm.txt", "w") as output:  ## creates new file but empty
        for arg in vars(ddpm_args):
            print(arg, getattr(ddpm_args, arg))
            output.write(str(arg) + "\t" + str(getattr(ddpm_args, arg)) + "\n")

    out_f = open(main_path + "output.txt",'w')

    if not os.path.exists(main_path):
        print('Creating directory at:', main_path, file=out_f)
        os.makedirs(main_path)
    
    print("Pytorch Version:", torch.__version__, file=out_f)
    print("Experiment: "+lsd_ebm_args.experiment, file=out_f)
    print("Main path: ", main_path, file=out_f)
    
    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = lsd_ebm_args.batch_size
    
    # diffusion 
    lsd_ebm_model = LSD_EBM_model()
    lebm_model = LEBM_model()
    ddpm_model = DDPM_model()

    lsd_ebm_model.load_model(base_model_path="./Vert_data/LSD-EBM_Vert_step_30")
    lebm_model.load_model(base_model_path="./Vert_data/xxx")
    ddpm_model.load_model(base_model_path="./Vert_data/xxx")

    z_noise = torch.randn((1, lsd_ebm_args.num_latents)).to(device) # generate the same noise

    x_noise = torch.randn((ddpm_args.img_size, ddpm_args.img_size, ddpm_args.img_size)).to(device) # generate the same noise
    

    lsd_ebm_generation = lsd_ebm_model.generation(z_noise)

    lebm_generation = lebm_model.generation(z_noise)

    ddpm_generation = ddpm_model.generation(x_noise)
