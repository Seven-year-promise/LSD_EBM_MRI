from __future__ import print_function, division
import os
import argparse
import time
import pickle
from torch.utils.data import DataLoader
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from brokenaxes import brokenaxes

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
#from get_model_ddpm import DDPM_model
#from get_model_ddpm import args as ddpm_args

import get_model_lsd_ebm as lsd_ebm_func

import get_model_lebm as lebm_func

from scipy.stats import norm, kurtosis


def wasserstein_2_1d(p, q):
    """
    :param data_a:
    :param data_b:
    :return ||m_1 - m_2||_2^2 + trace(c_1+c_2-2*(c_2^(0.5)*c_1*c_2^(0.5))^(0.5))
    """
    sign = 1
    mu_p = np.mean(p)
    mu_q = np.mean(q)
    c_p = np.cov(p)
    c_q = np.cov(q)
    return sign*((mu_p-mu_q)**2 + c_p + c_q -2*np.sqrt(np.sqrt(c_q)*c_p*np.sqrt(c_q)))


#%%Main
if  __name__ == "__main__" :   
    run_name = "compare_latent_lsdebm_lebm"
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
        
    #with open(main_path + "hyper_para_ddpm.txt", "w") as output:  ## creates new file but empty
    #    for arg in vars(ddpm_args):
    #        print(arg, getattr(ddpm_args, arg))
    #        output.write(str(arg) + "\t" + str(getattr(ddpm_args, arg)) + "\n")

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
    #ddpm_model = DDPM_model()

    lsd_ebm_model.load_model(base_model_path="./Vert_data/LSD-EBM_Vert_step_20")
    lebm_model.load_model(base_model_path="./Vert_data/LEBM_Vert_MCMC_step_20")
    #ddpm_model.load_model(base_model_path="./Vert_data/xxx")


    print("### Make predictions on test dataset", file=out_f)

    experiment_recon = 'recon4_augmheavy'
    experiment_recon_save = 'my_recon4_augmheavy'
    patients = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
    verts_recon = [1,2,3,4,5]
    testds_path = "./MRI_reconstructions/" 

    z_noise = torch.randn((1, 100)).to(device)

    hq_path  = os.path.join(testds_path, experiment_recon, '17_mri', '2_LQ.nii.gz') # 09-5, 12-4, 17-2
    

    hq_nib = nib.load(hq_path)
    hq_patch = hq_nib.get_fdata().astype("int8")
    print(hq_patch.max(), file=out_f)
    hq_patch[hq_patch!=0]=1
    print(hq_patch.min(), file=out_f)
    print(hq_patch.max(), file=out_f)
    full_patch = torch.from_numpy(hq_patch).view(1,1,128,128,128).to(device=device, dtype=torch.float)

    lebm_zs_1 = lebm_model.get_all_latents(full_patch)
    lebm_zs_2 = lebm_model.get_all_latents(full_patch)
    lebm_zs_3 = lebm_model.get_all_latents(full_patch)


    #  latent fot lsd run 1 
    lsd_z_diffusion_1, lsd_z_denoising_1 = lsd_ebm_model.get_all_latents(full_patch, step=20)
    lsd_z_diffusion_2, lsd_z_denoising_2 = lsd_ebm_model.get_all_latents(full_patch, step=20)
    lsd_z_diffusion_3, lsd_z_denoising_3 = lsd_ebm_model.get_all_latents(full_patch, step=20)

    latent_eval_lsdebm_fp_1 = []
    gaussianity_lsdebm_fp_1 = []
    var_lsdebm_fp_1 = []
    for i, z_d in enumerate(lsd_z_diffusion_1):
        z_d = z_d.to('cpu').detach().numpy()
        z_d = z_d[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), z_d)
        gau = kurtosis(z_d)
        var = np.var(z_d)
        latent_eval_lsdebm_fp_1.append([i, dis])
        gaussianity_lsdebm_fp_1.append([i, gau])
        var_lsdebm_fp_1.append([i, var])

    T_plus_1 = len(lsd_z_diffusion_1)
    latent_eval_lsdebm_bp_1 = []
    gaussianity_lsdebm_bp_1 = []
    var_lsdebm_bp_1 = []
    latent_eval_lsdebm_bp_1.append(latent_eval_lsdebm_fp_1[-1])
    gaussianity_lsdebm_bp_1.append(gaussianity_lsdebm_fp_1[-1])
    var_lsdebm_bp_1.append(var_lsdebm_fp_1[-1])
    for j, z_de in enumerate(lsd_z_denoising_1):
        z_de = z_de.to('cpu').detach().numpy()
        z_de = z_de[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), z_de)
        gau = kurtosis(z_de)
        var = np.var(z_de)
        latent_eval_lsdebm_bp_1.append([T_plus_1-j-2, dis])
        gaussianity_lsdebm_bp_1.append([T_plus_1-j-2,gau])
        var_lsdebm_bp_1.append([T_plus_1-j-2, var])

    #  latent fot lsd run 2
    latent_eval_lsdebm_fp_2 = []
    gaussianity_lsdebm_fp_2 = []
    var_lsdebm_fp_2 = []
    for i, z_d in enumerate(lsd_z_diffusion_2):
        z_d = z_d.to('cpu').detach().numpy()
        z_d = z_d[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), z_d)
        gau = kurtosis(z_d)
        var = np.var(z_d)
        latent_eval_lsdebm_fp_2.append([i, dis])
        gaussianity_lsdebm_fp_2.append([i,gau])
        var_lsdebm_fp_2.append([i, var])

    T_plus_1 = len(lsd_z_diffusion_2)
    latent_eval_lsdebm_bp_2 = []
    gaussianity_lsdebm_bp_2 = []
    var_lsdebm_bp_2 = []
    latent_eval_lsdebm_bp_2.append(latent_eval_lsdebm_fp_2[-1])
    gaussianity_lsdebm_bp_2.append(gaussianity_lsdebm_fp_2[-1])
    var_lsdebm_bp_2.append(var_lsdebm_fp_2[-1])
    for j, z_de in enumerate(lsd_z_denoising_2):
        z_de = z_de.to('cpu').detach().numpy()
        z_de = z_de[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), z_de)
        gau = kurtosis(z_de)
        var = np.var(z_de)
        latent_eval_lsdebm_bp_2.append([T_plus_1-j-2, dis])
        gaussianity_lsdebm_bp_2.append([T_plus_1-j-2,gau])
        var_lsdebm_bp_2.append([T_plus_1-j-2, var])

    #  latent fot lsd run 3
    latent_eval_lsdebm_fp_3 = []
    gaussianity_lsdebm_fp_3 = []
    var_lsdebm_fp_3 = []
    for i, z_d in enumerate(lsd_z_diffusion_3):
        z_d = z_d.to('cpu').detach().numpy()
        z_d = z_d[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), z_d)
        gau = kurtosis(z_d)
        var = np.var(z_d)
        latent_eval_lsdebm_fp_3.append([i, dis])
        gaussianity_lsdebm_fp_3.append([i,gau])
        var_lsdebm_fp_3.append([i, var])

    T_plus_1 = len(lsd_z_diffusion_3)
    latent_eval_lsdebm_bp_3 = []
    gaussianity_lsdebm_bp_3 = []
    var_lsdebm_bp_3 = []
    latent_eval_lsdebm_bp_3.append(latent_eval_lsdebm_fp_3[-1])
    gaussianity_lsdebm_bp_3.append(gaussianity_lsdebm_fp_3[-1])
    var_lsdebm_bp_3.append(var_lsdebm_fp_3[-1])
    for j, z_de in enumerate(lsd_z_denoising_3):
        z_de = z_de.to('cpu').detach().numpy()
        z_de = z_de[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), z_de)
        gau = kurtosis(z_de)
        var = np.var(z_de)
        latent_eval_lsdebm_bp_3.append([T_plus_1-j-2, dis])
        gaussianity_lsdebm_bp_3.append([T_plus_1-j-2,gau])
        var_lsdebm_bp_3.append([T_plus_1-j-2, var])

    #  latent for lebm
    
    latent_eval_lebm_1 = []
    gaussianity_lebm_1 = []
    var_lebm_1 = []
    for i, lebm_z in enumerate(lebm_zs_1):
        lebm_z = lebm_z.to('cpu').detach().numpy()
        lebm_z = lebm_z[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), lebm_z)
        gau = kurtosis(lebm_z)
        var = np.var(lebm_z)
        latent_eval_lebm_1.append([i, dis])
        gaussianity_lebm_1.append([i,gau])
        var_lebm_1.append([i, var])

    latent_eval_lebm_2 = []
    gaussianity_lebm_2 = []
    var_lebm_2 = []
    for i, lebm_z in enumerate(lebm_zs_2):
        lebm_z = lebm_z.to('cpu').detach().numpy()
        lebm_z = lebm_z[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), lebm_z)
        gau = kurtosis(lebm_z)
        var = np.var(lebm_z)
        latent_eval_lebm_2.append([i, dis])
        gaussianity_lebm_2.append([i,gau])
        var_lebm_2.append([i, var])

    latent_eval_lebm_3 = []
    gaussianity_lebm_3 = []
    var_lebm_3 = []
    for i, lebm_z in enumerate(lebm_zs_3):
        lebm_z = lebm_z.to('cpu').detach().numpy()
        lebm_z = lebm_z[0]
        dis = wasserstein_2_1d(z_noise.to('cpu').detach().numpy(), lebm_z)
        gau = kurtosis(lebm_z)
        var = np.var(lebm_z)
        latent_eval_lebm_3.append([i, dis])
        gaussianity_lebm_3.append([i,gau])
        var_lebm_3.append([i, var])


    latent_eval_lsdebm_fp_1 = np.array(latent_eval_lsdebm_fp_1)
    latent_eval_lsdebm_bp_1 = np.array(latent_eval_lsdebm_bp_1)
    gaussianity_lsdebm_fp_1 = np.array(gaussianity_lsdebm_fp_1)
    gaussianity_lsdebm_bp_1 = np.array(gaussianity_lsdebm_bp_1)
    var_lsdebm_fp_1 = np.array(var_lsdebm_fp_1)
    var_lsdebm_bp_1 = np.array(var_lsdebm_bp_1)
    
    latent_eval_lsdebm_fp_2 = np.array(latent_eval_lsdebm_fp_2)
    latent_eval_lsdebm_bp_2 = np.array(latent_eval_lsdebm_bp_2)
    gaussianity_lsdebm_fp_2 = np.array(gaussianity_lsdebm_fp_2)
    gaussianity_lsdebm_bp_2 = np.array(gaussianity_lsdebm_bp_2)
    var_lsdebm_fp_2 = np.array(var_lsdebm_fp_2)
    var_lsdebm_bp_2 = np.array(var_lsdebm_bp_2)

    latent_eval_lsdebm_fp_3 = np.array(latent_eval_lsdebm_fp_3)
    latent_eval_lsdebm_bp_3 = np.array(latent_eval_lsdebm_bp_3)
    gaussianity_lsdebm_fp_3 = np.array(gaussianity_lsdebm_fp_3)
    gaussianity_lsdebm_bp_3 = np.array(gaussianity_lsdebm_bp_3)
    var_lsdebm_fp_3 = np.array(var_lsdebm_fp_3)
    var_lsdebm_bp_3 = np.array(var_lsdebm_bp_3)

    latent_eval_lebm_1 = np.array(latent_eval_lebm_1)
    latent_eval_lebm_2 = np.array(latent_eval_lebm_2)
    latent_eval_lebm_3 = np.array(latent_eval_lebm_3)
    gaussianity_lebm_1 = np.array(gaussianity_lebm_1)
    gaussianity_lebm_2 = np.array(gaussianity_lebm_2)
    gaussianity_lebm_3 = np.array(gaussianity_lebm_3)
    var_lebm_1 = np.array(var_lebm_1)
    var_lebm_2 = np.array(var_lebm_2)
    var_lebm_3 = np.array(var_lebm_3)

    #  print(lsd_ebm_latent.to('cpu').detach().numpy())
    
    plt.plot(latent_eval_lsdebm_fp_1[:, 0], latent_eval_lsdebm_fp_1[:, 1], linewidth=2.0, linestyle="-", color="#FFBE7A", label="Diffusion process (LSD-EBM)")
    #plt.plot(latent_eval_lsdebm_fp_2[:, 0], latent_eval_lsdebm_fp_2[:, 1], linewidth=2.0, linestyle="-", color="#FA7F6F", label="Diffusion process (LSD-EBM) run 2")
    #plt.arrow(x = latent_eval_lsdebm_fp[14, 0], y=latent_eval_lsdebm_fp[14, 1], 
    #          dx= 3, dy=(latent_eval_lsdebm_fp[17, 1] - latent_eval_lsdebm_fp[14, 1]), lw=1)
    plt.plot(latent_eval_lsdebm_bp_1[:, 0], latent_eval_lsdebm_bp_1[:, 1], linewidth=2.0, linestyle="-.", color="#FFBE7A", label="Denoising process (LSD-EBM)")
    #plt.plot(latent_eval_lsdebm_bp_2[:, 0], latent_eval_lsdebm_bp_2[:, 1], linewidth=2.0, linestyle="-.", color="#FA7F6F", label="Denoising process (LSD-EBM) run 2")
    plt.plot(latent_eval_lebm_1[:, 0], latent_eval_lebm_1[:, 1], linewidth=2.0, linestyle="--", color="#8ECFC9", label="Latent of LEBM (run 1)")
    plt.plot(latent_eval_lebm_2[:, 0], latent_eval_lebm_2[:, 1], linewidth=2.0, linestyle="--", color="#54B345", label="Latent of LEBM (run 2)")
    plt.plot(latent_eval_lebm_3[:, 0], latent_eval_lebm_3[:, 1], linewidth=2.0, linestyle="--", color="#8983BF", label="Latent of LEBM (run 3)")
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("$L$2-Wasserstein distance", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()

    plt.legend(loc="best", fontsize=14)

    plt.xlim(-0.5,20.5)
    plt.ylim(-5.5,2000)

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Arial']


    plt.savefig(os.path.join(main_path, "latent3.png"))

    plt.clf()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    #fig.suptitle('Different runs of LSD-EBM')
    ax1.plot(gaussianity_lsdebm_bp_1[:, 0], gaussianity_lsdebm_bp_1[:, 1], linewidth=2.0, linestyle="-.", color="#FFBE7A", label="run 1")
    ax2.plot(gaussianity_lsdebm_bp_2[:, 0], gaussianity_lsdebm_bp_2[:, 1], linewidth=2.0, linestyle="-.", color="#c82423", label="run 2")
    ax3.plot(gaussianity_lsdebm_bp_3[:, 0], gaussianity_lsdebm_bp_3[:, 1], linewidth=2.0, linestyle="-.", color="#FA7F6F", label="run 3")
    #phase_1_min = np.min(gaussianity_lsdebm_bp_1[:, 1])
    #phase_2_min = np.min(gaussianity_lsdebm_bp_2[:, 1])
    #phase_3_min = np.min(gaussianity_lsdebm_bp_3[:, 1])

    #phase_1_max = np.max(gaussianity_lsdebm_bp_1[:, 1])
    #phase_2_max= np.max(gaussianity_lsdebm_bp_2[:, 1])
    #phase_3_max = np.max(gaussianity_lsdebm_bp_3[:, 1])

    #bax = brokenaxes(ylims=((phase_1_min-0.05, phase_1_max+0.05),
    #                        (phase_2_min-0.05, phase_2_max+0.05),
    #                        (phase_3_min-0.05, phase_3_max+0.05)), hspace=.05)
    #plt.plot(gaussianity_lsdebm_fp_1[:, 0], gaussianity_lsdebm_fp_1[:, 1], linewidth=2.0, linestyle="-", color="#FFBE7A", label="Diffusion process (LSD-EBM) run 1")
    #plt.plot(gaussianity_lsdebm_fp_2[:, 0], gaussianity_lsdebm_fp_2[:, 1], linewidth=2.0, linestyle="-", color="#c82423", label="Diffusion process (LSD-EBM) run 2")
    #plt.plot(gaussianity_lsdebm_fp_3[:, 0], gaussianity_lsdebm_fp_3[:, 1], linewidth=2.0, linestyle="-", color="#FA7F6F", label="Diffusion process (LSD-EBM) run 3")
    #plt.arrow(x = latent_eval_lsdebm_fp[14, 0], y=latent_eval_lsdebm_fp[14, 1], 
    #          dx= 3, dy=(latent_eval_lsdebm_fp[17, 1] - latent_eval_lsdebm_fp[14, 1]), lw=1)
    #bax.plot(gaussianity_lsdebm_bp_1[:, 0], gaussianity_lsdebm_bp_1[:, 1], linewidth=2.0, linestyle="-.", color="#FFBE7A", label="Denoising process (LSD-EBM) run 1")
    #bax.plot(gaussianity_lsdebm_bp_2[:, 0], gaussianity_lsdebm_bp_2[:, 1], linewidth=2.0, linestyle="-.", color="#c82423", label="Denoising process (LSD-EBM) run 2")
    #bax.plot(gaussianity_lsdebm_bp_3[:, 0], gaussianity_lsdebm_bp_3[:, 1], linewidth=2.0, linestyle="-.", color="#FA7F6F", label="Denoising process (LSD-EBM) run 3")
    #plt.plot(gaussianity_lebm_1[:, 0], gaussianity_lebm_1[:, 1], linewidth=2.0, linestyle="--", color="#8ECFC9", label="Latent of LEBM (run 1)")
    #plt.plot(gaussianity_lebm_2[:, 0], gaussianity_lebm_2[:, 1], linewidth=2.0, linestyle="--", color="#54B345", label="Latent of LEBM (run 2)")
    #plt.plot(gaussianity_lebm_3[:, 0], gaussianity_lebm_3[:, 1], linewidth=2.0, linestyle="--", color="#8983BF", label="Latent of LEBM (run 3)")
    fig.supxlabel("Steps", fontsize=14)
    fig.supylabel("Kurtosis of latent variable", fontsize=14)
    #ax1.set_xlabel("Steps", fontsize=14)
    #ax2.set_xlabel("Steps", fontsize=14)
    #ax3.set_xlabel("Steps", fontsize=14)
    #ax1.set_ylabel("Kurtosis of latent variable", fontsize=14)
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax2.xaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)
    ax3.xaxis.set_tick_params(labelsize=14)
    ax3.yaxis.set_tick_params(labelsize=14)
    
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(5))
    
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(5))
    
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(5))
    
    

    #plt.legend(loc="best", fontsize=14)
    ax1.legend(loc="best", fontsize=14)
    ax2.legend(loc="best", fontsize=14)
    ax3.legend(loc="best", fontsize=14)

    #plt.xlim(-0.5,20.5)
    #plt.ylim(-0.3,0.5)

    #ax1.rcParams['font.family'] = 'DeJavu Serif'
    #ax1.rcParams['font.serif'] = ['Arial']
    #ax2.rcParams['font.family'] = 'DeJavu Serif'
    #ax2.rcParams['font.serif'] = ['Arial']
    #ax3.rcParams['font.family'] = 'DeJavu Serif'
    #ax3.rcParams['font.serif'] = ['Arial']

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Arial']

    plt.tight_layout()
    plt.savefig(os.path.join(main_path, "gaussinity_lsd_ebm_latent3.png"))


    plt.clf()

    plt.plot(var_lsdebm_fp_1[:, 0], var_lsdebm_fp_1[:, 1], linewidth=2.0, linestyle="-", color="#FFBE7A", label="Diffusion process (run 1)")
    plt.plot(var_lsdebm_fp_2[:, 0], var_lsdebm_fp_2[:, 1], linewidth=2.0, linestyle="-", color="#c82423", label="Diffusion process (run 2)")
    plt.plot(var_lsdebm_fp_3[:, 0], var_lsdebm_fp_3[:, 1], linewidth=2.0, linestyle="-", color="#FA7F6F", label="Diffusion process (run 3)")
    #plt.arrow(x = latent_eval_lsdebm_fp[14, 0], y=latent_eval_lsdebm_fp[14, 1], 
    #          dx= 3, dy=(latent_eval_lsdebm_fp[17, 1] - latent_eval_lsdebm_fp[14, 1]), lw=1)
    plt.plot(var_lsdebm_bp_1[:, 0], var_lsdebm_bp_1[:, 1], linewidth=2.0, linestyle="-.", color="#FFBE7A", label="Denoising process (run 1)")
    plt.plot(var_lsdebm_bp_2[:, 0], var_lsdebm_bp_2[:, 1], linewidth=2.0, linestyle="-.", color="#c82423", label="Denoising process (run 2)")
    plt.plot(var_lsdebm_bp_3[:, 0], var_lsdebm_bp_3[:, 1], linewidth=2.0, linestyle="-.", color="#FA7F6F", label="Denoising process (run 3)")
    #plt.plot(gaussianity_lebm_1[:, 0], gaussianity_lebm_1[:, 1], linewidth=2.0, linestyle="--", color="#8ECFC9", label="Latent of LEBM (run 1)")
    #plt.plot(gaussianity_lebm_2[:, 0], gaussianity_lebm_2[:, 1], linewidth=2.0, linestyle="--", color="#54B345", label="Latent of LEBM (run 2)")
    #plt.plot(gaussianity_lebm_3[:, 0], gaussianity_lebm_3[:, 1], linewidth=2.0, linestyle="--", color="#8983BF", label="Latent of LEBM (run 3)")
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Variance of latent variable", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()

    plt.legend(loc="best", fontsize=14)

    plt.xlim(-0.5,20.5)
    #plt.ylim(-0.3,0.5)

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Arial']


    plt.savefig(os.path.join(main_path, "var_lsd_ebm_latent3.png"))


    plt.clf()
    #plt.plot(gaussianity_lsdebm_fp[:, 0], gaussianity_lsdebm_fp[:, 1], linewidth=2.0, linestyle="-", color="#FFBE7A", label="Diffusion process (LSD-EBM)")
    #plt.arrow(x = latent_eval_lsdebm_fp[14, 0], y=latent_eval_lsdebm_fp[14, 1], 
    #          dx= 3, dy=(latent_eval_lsdebm_fp[17, 1] - latent_eval_lsdebm_fp[14, 1]), lw=1)
    #plt.plot(gaussianity_lsdebm_bp[:, 0], gaussianity_lsdebm_bp[:, 1], linewidth=2.0, linestyle="-", color="#FA7F6F", label="Denoising process (LSD-EBM)")
    plt.plot(gaussianity_lebm_1[:, 0], gaussianity_lebm_1[:, 1], linewidth=2.0, linestyle="--", color="#8ECFC9", label="run 1")
    plt.plot(gaussianity_lebm_2[:, 0], gaussianity_lebm_2[:, 1], linewidth=2.0, linestyle="--", color="#54B345", label="run 2")
    plt.plot(gaussianity_lebm_3[:, 0], gaussianity_lebm_3[:, 1], linewidth=2.0, linestyle="--", color="#8983BF", label="run 3")
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Kurtosis of latent variable", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()

    plt.legend(loc="best", fontsize=14)

    plt.xlim(-0.5,20.5)
    #plt.ylim(-5.5,1800)

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Arial']


    plt.savefig(os.path.join(main_path, "gaussinity_lebm_latent3.png"))

    plt.clf()
    #plt.plot(gaussianity_lsdebm_fp[:, 0], gaussianity_lsdebm_fp[:, 1], linewidth=2.0, linestyle="-", color="#FFBE7A", label="Diffusion process (LSD-EBM)")
    #plt.arrow(x = latent_eval_lsdebm_fp[14, 0], y=latent_eval_lsdebm_fp[14, 1], 
    #          dx= 3, dy=(latent_eval_lsdebm_fp[17, 1] - latent_eval_lsdebm_fp[14, 1]), lw=1)
    #plt.plot(gaussianity_lsdebm_bp[:, 0], gaussianity_lsdebm_bp[:, 1], linewidth=2.0, linestyle="-", color="#FA7F6F", label="Denoising process (LSD-EBM)")
    plt.plot(var_lebm_1[:, 0], var_lebm_1[:, 1], linewidth=2.0, linestyle="--", color="#8ECFC9", label="run 1")
    plt.plot(var_lebm_2[:, 0], var_lebm_2[:, 1], linewidth=2.0, linestyle="--", color="#54B345", label="run 2")
    plt.plot(var_lebm_3[:, 0], var_lebm_3[:, 1], linewidth=2.0, linestyle="--", color="#8983BF", label="run 3")
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Variance of latent variable", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.tight_layout()

    plt.legend(loc="best", fontsize=14)

    plt.xlim(-0.5,20.5)
    #plt.ylim(-5.5,1800)

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Arial']


    plt.savefig(os.path.join(main_path, "var_lebm_latent3.png"))

    """
    #plt.clf()

    plt.plot(latent_eval_lsdebm_bp[:, 0], latent_eval_lsdebm_bp[:, 1], linewidth=2.0, linestyle="-", color="#FA7F6F", label="Reverse process (LSD-EBM)")
    plt.xlabel("Steps (MCMC or diffusion)", fontname="Arial", fontsize=16)
    plt.ylabel("$L$2-Wasserstein distance", fontname="Arial", fontsize=16)

    #plt.legend(loc="best")
    #plt.legend(fontsize=16)
    plt.xticks(fontname="Arial", fontsize=16)
    plt.yticks(fontname="Arial", fontsize=16)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig(os.path.join(main_path, "latent_bp3.png"))

    """


