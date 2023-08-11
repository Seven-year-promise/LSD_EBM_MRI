from __future__ import print_function, division
import os
import argparse
import time
import pickle
from torch.utils.data import DataLoader
import numpy as np
import scipy
import csv
from time import process_time

import torch
import torch.optim as optim
import nibabel as nib
import torch.nn as nn
from medpy.metric.binary import dc

from get_model_lsd_ebm import LSD_EBM_model
from get_model_lsd_ebm import args as lsd_ebm_args

from get_model_lebm import LEBM_model
from get_model_lebm import args as lebm_args

from get_model_ddpm import DDPM_model
import get_model_ddpm

from get_model_vae import VAE_model

from utils import calc_VolumetricSimilarity, calc_SimpleHausdorffDistance, calc_Sensitivity_Sets, calc_Specificity_Sets
import pandas as pd

def DiceCoeff(pred, gt):
     pred = pred.to('cpu').numpy()
     gt = gt.to('cpu').numpy()
     
     #if gt is all zero (use inverse to count)
     if np.count_nonzero(gt) == 0:
         gt = gt+1
         pred = 1-pred
         
     dice_score = dc(pred.astype(np.int),gt.astype(np.int)) 
     return dice_score

#%%Main
if  __name__ == "__main__" :   
   
    main_path="./comp_results/" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    #dice_recon_h_csv_file = open(on.path.join(main_path, 'dice_h_recon.csv'), 'w', newline='')
    #dice_recon_h_csv_writer = csv.writer(dice_recon_h_csv_file, delimiter=';',
    #                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    out_f = open(main_path + "processing_time.txt",'w')

    lsd_ebm_model = LSD_EBM_model()
    lebm_model = LEBM_model()
    ddpm_model = DDPM_model()
    vae_model = VAE_model()

    lsd_dice_recon_l = {}
    lsd_dice_recon_h = {}
    lsd_vs_recon_l = {}  # volumetric similarity
    lsd_vs_recon_h = {}
    lsd_hd_recon_l = {} # Hausdordd distance
    lsd_hd_recon_h = {}
    lsd_sen_recon_l = {} # Sensitivity via Sets   
    lsd_sen_recon_h = {}
    lsd_spec_recon_l = {} # Specificity  
    lsd_spec_recon_h = {}


    lebm_dice_recon_l = {}
    lebm_dice_recon_h = {}
    lebm_vs_recon_l = {} # volumetric similarity
    lebm_vs_recon_h = {}
    lebm_hd_recon_l = {} # Hausdordd distance
    lebm_hd_recon_h = {}
    lebm_sen_recon_l = {} # Sensitivity via Sets   
    lebm_sen_recon_h = {}
    lebm_spec_recon_l = {} # Specificity  
    lebm_spec_recon_h = {}
    
    print("### Make predictions on test dataset")

    experiment_recon = 'recon4_augmheavy'
    experiment_recon_save = 'my_recon4_augmheavy'
    patients = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
    verts_recon = [1,2,3,4,5]
    testds_path = "./MRI_reconstructions/" 

    steps = [2, 15, 20]
    lsd_base_model_path = "./Vert_data/LSD-EBM_Vert_step_"
    lebm_base_model_path = "./Vert_data/LEBM_Vert_MCMC_step_"
    ddpm_base_model_path = "./DDPM_Vert/logs/DDPM_Vert"
    vae_base_model_path = "./Vert_data/VAE_Vert_01"

    step= 20
    
    print("LOAD LSD EBM MODEL EP ", lsd_base_model_path+str(step), lsd_ebm_args.load_from_ep, file=out_f)
    lsd_ebm_model.load_model(lsd_base_model_path+str(step))
    print("LOAD LEBM MODEL EP ", lebm_base_model_path+str(step), lebm_args.load_from_ep, file=out_f)
    lebm_model.load_model(lebm_base_model_path+str(step))
    print("LOAD DDPM", ddpm_base_model_path, file=out_f)
    #ddpm_model.load_model(ddpm_base_model_path)
    print("LOAD VAE", vae_base_model_path, file=out_f)
    vae_model.load_model(vae_base_model_path)

    patient = '17'
    print("## Patient {}".format(patient), file=out_f)
    idx = 2
    print("# Vertebra {}".format(verts_recon[idx]), file=out_f)

    # load the high-quality data
    
    hq_path  = os.path.join(testds_path, experiment_recon,patient+'_mri',str(verts_recon[idx])+'_HQ.nii.gz')


    hq_nib = nib.load(hq_path)
    hq_patch = hq_nib.get_fdata().astype("int8")
    print(hq_patch.max(), file=out_f)
    hq_patch[hq_patch!=0]=1
    print(hq_patch.min(), file=out_f)
    print(hq_patch.max(), file=out_f)
    full_h_patch = torch.from_numpy(hq_patch).view(1,1,128,128,128).to(device=device, dtype=torch.float)

    t1 = process_time() 
    h_pred_patch_lsd = lsd_ebm_model.reconstruction(full_h_patch, step=step)

    t2 = process_time() 

    h_pred_patch_lebm = lebm_model.reconstruction(full_h_patch)

    t3 = process_time() 

    h_pred_patch_vae = vae_model.reconstruction(full_h_patch)

    t4 = process_time() 

    print("step ", step, "LSD-EBM processing time (s)", t2-t1, file=out_f)
    print("step ", step, "LEBM processing time (s)", t3-t2, file=out_f)
    print("VAE processing time (s)", t4-t3, file=out_f)



