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
from model_Unet_lebm import ReconNet, E_func, GenModel_vert, sample_langevin_prior_z, sample_langevin_post_z, Buff
from dataset_vae import CSI_Dataset
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


#%%Main
if  __name__ == "__main__" :   
    
    fixed_images = ['L1_verse033_seg.nii.gz', 'L1_verse265_seg.nii.gz', 'L2_verse145_seg.nii.gz',  'L3_verse082_seg.nii.gz']
    
    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')
    
    # Trianing
    parser.add_argument('--num_epochs',                         default=201, type=int, help='Number of epochs')   # 0                   
    parser.add_argument('--batch_size',                         default=2, type=int,help='Batch Size')
    parser.add_argument('--ebm_lr', type=float,                 default=0.0001,help='learning rate of the EMB network')
    parser.add_argument('--gen_lr', type=float,                 default=0.0001,help='learning rate of the generation network')
    parser.add_argument('--load_from_ep',                       default=None, type=int, help='checkpoint you want to load for the models')
    parser.add_argument('--epoch_start',                        default=0, type=int, help='epoch you want to start from') #200
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
    parser.add_argument('--pr_mcmc_steps_tr',                   default=30, type=int,help='Number of mcmc steps sampling from prior for training') #60
    parser.add_argument('--pr_mcmc_steps_val',                  default=30, type=int,help='Number of mcmc steps sampling from prior for validation') #60
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
   
    run_name = "LEBM_Vert_MCMC_step_" + str(args.pr_mcmc_steps_tr) 
    main_path="./../Vert_data/" + run_name + "/" 

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    with open(main_path + "hyper_para.txt", "w") as output:  ## creates new file but empty
        for arg in vars(args):
            print(arg, getattr(args, arg))
            output.write(str(arg) + "\t" + str(getattr(args, arg)) + "\n")

    out_f = open(main_path + "output.txt",'w')

    if not os.path.exists(main_path):
        print('Creating directory at:', main_path, file=out_f)
        os.makedirs(main_path)
    
    print("Pytorch Version:", torch.__version__, file=out_f)
    print("Experiment: "+args.experiment, file=out_f)
    print(args, file=out_f)
    print("Main path: ", main_path, file=out_f)
    
    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    
    # Initialize networks and buffer
    e_func    = E_func(args.num_latents, args.ebm_inner_dim, args.ebm_num_layers, batchnorm=args.prior_bn).to(device)
    gen_model = GenModel_vert(args.num_channels, args.num_latents).to(device)


    def sample_p_0(n, sig=args.prior_var):
        return sig * torch.randn(*[n, args.num_latents])
        
    buff = None
    if args.use_samp_buff:
        buff = Buff(sample_p_0(args.batch_size), args.samp_buff_size)
        buff.add_samp(sample_p_0(5))
    
    # use multiple gpus
    n_gpus=torch.cuda.device_count()
        
    #load models if requested
    if args.load_from_ep is not None:
        e_func.load_state_dict(torch.load(os.path.join(main_path,'experiments', 'efunc_'+str(args.load_from_ep)+'.pth')))
        print("LOAD EBM MODEL EP ", args.load_from_ep, file=out_f)
        gen_model.load_state_dict(torch.load(os.path.join(main_path,'experiments', 'genmodel_'+str(args.load_from_ep)+'.pth')))
        print("LOAD GENERATION MODEL", args.load_from_ep, file=out_f)
        if args.use_samp_buff: 
            buff = torch.load(os.path.join(main_path,'experiments', 'buffer_'+str(args.load_from_ep)+'.pth'))
            print("LOAD SAMPLE BUFFER", args.load_from_ep, file=out_f)
            print("SIZE ", buff.buff.size()[0], file=out_f)
        
    # optimizesr
    mse = nn.MSELoss(reduction='sum')
    bce = nn.BCELoss(reduction='sum')
    if args.recon_loss=='mse':
        recon_loss = mse
    else:
        recon_loss = bce

    optE = torch.optim.Adam(e_func.parameters(),    lr=args.ebm_lr) 
    optG = torch.optim.Adam(gen_model.parameters(), lr=args.gen_lr) 
    if args.ebm_dyn_lr != None:
        schedulerE = torch.optim.lr_scheduler.ExponentialLR(optE, gamma=args.ebm_dyn_lr)
    if args.gen_dyn_lr != None:
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=args.gen_dyn_lr)

    train_root      = os.path.join(main_path, "..",  args.train_set)
    validation_root = os.path.join(main_path, "..",  args.validation_set)
    test_root       = os.path.join(main_path , "..", args.test_set)
    visu_path=os.path.join(main_path,'visu',args.experiment)
    
    
    fixed_full, fixed_body = load_fixed(fixed_images, test_root)
    for idx in range(len(fixed_full)):
        fixed_full[idx] = fixed_full[idx].to(device=device, dtype=torch.float)
        fixed_body[idx] = fixed_body[idx].to(device=device, dtype=torch.float)
       
    imgs_list_train      =os.listdir(os.path.join(train_root,'full'))
    imgs_list_test       =os.listdir(os.path.join(test_root,'full'))
    imgs_list_validation =os.listdir(os.path.join(validation_root,'full'))
    
    list_length_train       =len(imgs_list_train)
    list_length_validation  =len(imgs_list_validation)
    list_length_test        =len(imgs_list_test)
    print("Training set size: "  , list_length_train, file=out_f) 
    print("Validation set size: ", list_length_validation, file=out_f) 
    print("Test set size: "      , list_length_test, file=out_f) 
    
    #create variables as dictionaries
    dataset={}
    loader={}
    loss_e_dic={}
    loss_recon_dic={}
    v_dice_dic={}
    v_dice_proc_dic={}
    v_fid_dic={}
    
    
    for i, mode in enumerate(modes+['test']):
        if mode == 'train':
            dataset[mode]=CSI_Dataset(train_root, mode=mode)
            if batch_size>list_length_train:
                loader[mode]=DataLoader(dataset[mode], batch_size=list_length_train, shuffle=True, drop_last=True)
            else:
                loader[mode]=DataLoader(dataset[mode], batch_size=batch_size, shuffle=True, drop_last=True)
        elif mode == 'validation':
            dataset[mode]=CSI_Dataset(validation_root, mode=mode)
            if batch_size>list_length_validation:
                loader[mode]=DataLoader(dataset[mode], batch_size=list_length_validation, shuffle=True, drop_last=True)
            else:
                loader[mode]=DataLoader(dataset[mode], batch_size=batch_size, shuffle=True, drop_last=True)
        elif mode == 'test' and args.test_perf == True:
            dataset[mode]=CSI_Dataset(test_root, mode=mode)
            loader[mode]=DataLoader(dataset[mode], batch_size=1, shuffle=False, drop_last=True)
        
        loss_e_dic[mode]=[]
        loss_recon_dic[mode]=[]
        v_dice_dic[mode]=[]
        v_dice_proc_dic[mode]=[]
        v_fid_dic[mode]=[]
                        
    
    # Start Training/testing
    for epoch in range(args.epoch_start,args.epoch_start+args.num_epochs):

        start_time = time.time()
        
        epoch_v_energy_real = {}
        epoch_v_energy_noise = {}
        epoch_v_energy_samp = {}
        epoch_v_loss_e = {}
        epoch_v_loss_recon = {}
        epoch_v_dice = {}
        epoch_v_dice_proc = {}
        epoch_v_fid = {}
        
                
        for mode in modes:
            if mode != 'train' and epoch%args.ep_betw_val!=0:
                continue

            if mode=='train':
                e_func.train()
                gen_model.train()
            else:
                e_func.eval()
                gen_model.eval()

            
            epoch_v_energy_real[mode]=[]
            epoch_v_energy_noise[mode]=[]
            epoch_v_energy_samp[mode]=[]
            epoch_v_loss_e[mode]=[]
            epoch_v_loss_recon[mode]=[]
            epoch_v_dice[mode]=[]
            epoch_v_dice_proc[mode]=[]
            epoch_v_fid[mode]=[]


            for i, data in enumerate(loader[mode], 0):
                torch.cuda.empty_cache()
                
                print("mem alloc: ", torch.cuda.memory_allocated(0)/1000000, file=out_f)
                print("mem max: "  , torch.cuda.get_device_properties(0).total_memory/1000000, file=out_f)
                
                img_name, full_patch, body_patch = data
                
                full_patch = full_patch.to(device=device, dtype=torch.float)
                local_bs = full_patch.size()[0]
                '''
                if i == 5:
                    for g in optimizerVae.param_groups:
                        g['lr'] = 0.0001
                ''' 
                ############################
                # (1) Update VAE network
                ############################
                    
                if mode=='train':
                    optE.zero_grad()
                    optG.zero_grad()
                
                
                ### START LATENT EBM ###

                ### 1. Sample z from prior and posterior
                if mode=='train':
                    if args.use_samp_buff:
                        # z_g
                        num_new = np.random.binomial(local_bs, 0.25)
                        z_g_0_new  = sample_p_0(num_new)
                        z_g_0_buff = buff.get_samp(local_bs-num_new)
                        z_g_0      = torch.cat([z_g_0_new, z_g_0_buff]).to(device)
                        # z_e
                        num_new = np.random.binomial(local_bs, 0.25)
                        z_e_0_new  = sample_p_0(num_new)
                        z_e_0_buff = buff.get_samp(local_bs-num_new)
                        z_e_0      = torch.cat([z_e_0_new, z_e_0_buff]).to(device)
                        
                        z_g = sample_langevin_post_z(z_g_0, full_patch, gen_model, e_func, args.po_mcmc_steps_tr, args.po_mcmc_step_size, args.prior_var , args.po_mcmc_noise_var, args.gen_var, args.po_mcmc_step_gamma)
                        z_e = sample_langevin_prior_z(z_e_0, e_func, args.pr_mcmc_steps_tr, args.pr_mcmc_step_size, args.prior_var, args.pr_mcmc_noise_var, args.pr_mcmc_step_gamma)
                        
                        buff.add_samp(torch.cat([z_g, z_e]).cpu())
                    else:
                        z_g_0 = sample_p_0(local_bs).to(device)  
                        z_e_0 = sample_p_0(local_bs).to(device) 
                        z_g = sample_langevin_post_z(z_g_0, full_patch, gen_model, e_func, args.po_mcmc_steps_tr, args.po_mcmc_step_size, args.prior_var , args.po_mcmc_noise_var, args.gen_var, args.po_mcmc_step_gamma)
                        z_e = sample_langevin_prior_z(z_e_0, e_func, args.pr_mcmc_steps_tr, args.pr_mcmc_step_size, args.prior_var, args.pr_mcmc_noise_var, args.pr_mcmc_step_gamma)
                else:
                    z_g_0 = sample_p_0(local_bs).to(device)  
                    z_e_0 = sample_p_0(local_bs).to(device) 
                    z_g = sample_langevin_post_z(z_g_0, full_patch, gen_model, e_func, args.po_mcmc_steps_val, args.po_mcmc_step_size, args.prior_var, args.po_mcmc_noise_var, args.gen_var, args.po_mcmc_step_gamma)
                    z_e = sample_langevin_prior_z(z_e_0, e_func, args.pr_mcmc_steps_val, args.pr_mcmc_step_size, args.prior_var, args.pr_mcmc_noise_var, args.pr_mcmc_step_gamma)

                ### 2. Learn generator
                recon_patch = gen_model.forward(z_g)
                loss_g = recon_loss(recon_patch, full_patch) / local_bs
                #loss_g.backward()

                ### 3. Learn prior EBM
                en_samp_v = e_func.forward(z_e.detach()) 
                en_real_v = e_func.forward(z_g.detach())
                en_samp = en_samp_v.mean()
                en_real = en_real_v.mean()
                
                loss_e = en_real - en_samp + args.EBM_reg * (en_samp_v**2 + en_real_v**2).mean() 
                #loss_e.backward()
                
                if mode=='train':
                    torch.autograd.backward(loss_g, inputs=list(gen_model.parameters()), retain_graph=True)
                    torch.autograd.backward(loss_e, inputs=list(e_func.parameters()))
                    optE.step() # optim. step
                    optG.step() # optim. step
                    
                    
                with torch.no_grad(): 
                    pred_patch = torch.round(recon_patch).detach() 
                    dice = DiceCoeff(pred_patch, full_patch.detach()) #Note that the recon itself is not binary
                    dice_proc = 0.
                    
                    # Energies
                    en_noise = e_func.forward(z_e_0).mean()
                    
                    # FID score
                    mu1   = z_g.view(-1, args.num_latents).mean(dim=0).cpu()
                    mu2   = z_e.view(-1, args.num_latents).mean(dim=0).cpu()
                    #cov1  = z_g.view(-1, args.num_latents).cov().cpu()
                    #cov2  = z_e.view(-1, args.num_latents).cov().cpu()

                    cov1  = cov(z_g.view(-1, args.num_latents)).cpu()# z_inf.view(-1, args.num_latents).cov().cpu()
                    cov2  = cov(z_e.view(-1, args.num_latents)).cpu()
                    fid_score            = (mu1 - mu2).dot(mu1 - mu2) + np.trace(cov1 + cov2 - 2*scipy.linalg.sqrtm(cov1*cov2))

                print('[%d/%d][%d/%d]\ten_real: %.4f\ten_noise: %.4f\ten_samp: %.4f\tE_l: %.4f\tMSE_l: %.4f\tDice: %.4f\tDiceProc: %.4f\tFID: %.4f \n'% (epoch, args.num_epochs, i, len(loader[mode]), en_real.item(), en_noise.item(), en_samp.item(), loss_e.item(), loss_g.item(), dice, dice_proc, fid_score), file=out_f)
                
                epoch_v_energy_real[mode].append(en_real.item())
                epoch_v_energy_noise[mode].append(en_noise.item())
                epoch_v_energy_samp[mode].append(en_samp.item())
                epoch_v_loss_e[mode].append(loss_e.item())
                epoch_v_loss_recon[mode].append(loss_g.item())
                epoch_v_dice[mode].append(dice)
                epoch_v_dice_proc[mode].append(dice_proc)
                epoch_v_fid[mode].append(fid_score)
                
           
                if args.save_visu and (epoch%20==0):
                    patch_path=os.path.join(visu_path,str(epoch),mode,'patch_'+str(i)+'_'+img_name[0][:-7])
                    try:
                        os.makedirs(patch_path)
                    except OSError:
                        None
                    # save reconstruction
                    with torch.no_grad():
                        nib.save(nib.Nifti1Image(np.squeeze(pred_patch.to('cpu').numpy()[0]),affine=None),patch_path+ '/recon_patch.nii.gz')
                        nib.save(nib.Nifti1Image(np.squeeze(full_patch.to('cpu').numpy()[0]),affine=None), patch_path+ '/full_patch.nii.gz')
                    del pred_patch # free memory on GPU
                    # save generation (if in validation mode)
                    if mode=='validation':
                        gen_patch = torch.round(gen_model.forward(z_e))
                        with torch.no_grad():
                            nib.save(nib.Nifti1Image(np.squeeze(gen_patch.to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/gen_patch.nii.gz')
                        del gen_patch  # free memory on GPU
                    # save loss information
                    with open(os.path.join(visu_path,str(epoch),mode,"loss.txt"), "a") as f:
                        f.write("\nPatch {}: en_real: {:.6} en_noise: {:.6} en_samp: {:.6} Eloss: {:.6} MSEloss: {:.6}   VaeDice: {:.6}  FID: {:.6}".format(str(i), en_real.item(), en_noise.item(), en_samp.item(), loss_e.item(), loss_g.item(), dice, fid_score))
                    with open(os.path.join(visu_path,str(epoch),mode,"loss.csv"), "a") as f:
                        f.write("{};{};{};{};{};{};{};{}".format(str(i), en_real.item(), en_noise.item(), en_samp.item(), loss_e.item(), loss_g.item(), dice, fid_score))

                
            loss_e_dic[mode].append(epoch_v_loss_e[mode])
            loss_recon_dic[mode].append(epoch_v_loss_recon[mode])
            v_dice_dic[mode].append(epoch_v_dice[mode])
            v_dice_proc_dic[mode].append(epoch_v_dice_proc[mode])
            v_fid_dic[mode].append(epoch_v_fid[mode])
            
            
            avg_v_loss_e = sum(epoch_v_loss_e[mode]) / len(epoch_v_loss_e[mode])
            avg_v_loss_recon = sum(epoch_v_loss_recon[mode]) / len(epoch_v_loss_recon[mode])
            avg_v_dice = sum(epoch_v_dice[mode]) / len(epoch_v_dice[mode])
            avg_v_dice_proc = sum(epoch_v_dice_proc[mode]) / len(epoch_v_dice_proc[mode])
            avg_v_fid = sum(epoch_v_fid[mode]) / len(epoch_v_fid[mode])
            avg_v_en_real = sum(epoch_v_energy_real[mode]) / len(epoch_v_energy_real[mode])
            avg_v_en_noise = sum(epoch_v_energy_noise[mode]) / len(epoch_v_energy_noise[mode])
            avg_v_en_samp = sum(epoch_v_energy_samp[mode]) / len(epoch_v_energy_samp[mode])
            
            print('\n{} Epoch: {} \t  En real: {:.4f} \t  En noise: {:.4f} \t  En samp: {:.4f} \t  ELoss: {:.4f} \t  MSELoss: {:.4f} \t VaeDice: {:.4} \t VaeDiceProc: {:.4} \t FID: {:.4f}%\n\n'.format(mode,epoch, avg_v_en_real, avg_v_en_noise, avg_v_en_samp, avg_v_loss_e, avg_v_loss_recon, avg_v_dice, avg_v_dice_proc, avg_v_fid), file=out_f)
            
            #if args.save_visu and (epoch%50==0):
            with open(os.path.join(visu_path,"epoch_loss.txt"), "a") as f:
                f.write("Epoch {}: {}  En real: {}  En noise: {}  En samp: {}  VlossE: {}  VlossMSE: {}  VaeDice:{}  VaeDiceProc:{}  FID:{} \n".format(epoch, mode, avg_v_en_real, avg_v_en_noise, avg_v_en_samp, avg_v_loss_e, avg_v_loss_recon, avg_v_dice, avg_v_dice_proc, avg_v_fid))
                if mode == 'validation':
                    print("\n")
            with open(os.path.join(visu_path,"epoch_loss.csv"), "a") as f:
                f.write("{};{};{};{};{};{};{};{};{};{} \n".format(epoch, mode, avg_v_en_real, avg_v_en_noise, avg_v_en_samp, avg_v_loss_e, avg_v_loss_recon, avg_v_dice, avg_v_dice_proc, avg_v_fid))
            
        """    
        if args.save_visu and (epoch%20==0):
            #with torch.no_grad():
            for idx in range(len(fixed_full)):
                print("Fixed image: " + fixed_images[idx])
                patch_path=os.path.join(visu_path,str(epoch),'fixed',fixed_images[idx][:-7])
                try:
                    os.makedirs(patch_path)
                except OSError:
                    None
                    
                z_g_0 = sample_p_0(local_bs).to(device) 
                z_g = sample_langevin_post_z(z_g_0, fixed_full[idx], gen_model, e_func, args.po_mcmc_steps_val, args.po_mcmc_step_size, args.prior_var, args.po_mcmc_noise_var, args.gen_var)
                fixed_recon = gen_model.forward(z_g)
                fixed_pred = torch.round(fixed_recon).detach()
                fixed_dice = DiceCoeff(fixed_pred, fixed_full[idx].detach())
                fixed_dice_proc = DiceCoeffProc(fixed_pred,  fixed_full[idx].detach(),  fixed_body[idx].detach())
                nib.save(nib.Nifti1Image(np.squeeze(fixed_pred.to('cpu').detach().numpy()),affine=None),patch_path+ '/recon_patch.nii.gz')
                nib.save(nib.Nifti1Image(np.squeeze(fixed_full[idx].to('cpu').detach().numpy()),affine=None), patch_path+ '/full_patch.nii.gz')   
                with open(os.path.join(visu_path,str(epoch),'fixed',"recon_loss.txt"), "a") as f:
                    f.write(fixed_images[idx] + "\tDice:{}\tDiceProc:{} \n".format(fixed_dice, fixed_dice_proc))
                with open(os.path.join(visu_path,str(epoch),'fixed',"recon_loss.csv"), "a") as f:
                    f.write("{};{};{};{} \n".format(epoch, mode, fixed_dice, fixed_dice_proc))
        """

        #Saving current model
        if epoch%20==0:
            lebm_model_path = os.path.join(main_path,'experiments')
            if not os.path.exists(lebm_model_path):
                print('Creating directory at:', lebm_model_path, file=out_f)
                os.makedirs(lebm_model_path)
            torch.save(e_func.state_dict(),os.path.join(lebm_model_path,'efunc_'+str(epoch)+'.pth'))
            torch.save(gen_model.state_dict(),os.path.join(lebm_model_path,'genmodel_'+str(epoch)+'.pth'))
            if args.use_samp_buff: torch.save(buff,os.path.join(lebm_model_path,'buffer_'+str(epoch)+'.pth'))
            with open(os.path.join(lebm_model_path,args.experiment+str(epoch)+".p"), 'wb') as f:
                pickle.dump([loss_e_dic, loss_recon_dic, v_dice_dic, v_dice_proc_dic, v_fid_dic],f)
        
        if args.ebm_dyn_lr != None:
            schedulerE.step()
        if args.gen_dyn_lr != None:
            schedulerG.step()
       
        print("--- %s seconds ---" % (time.time() - start_time), file=out_f)

    
    # TEST PERFORMANCE
    if args.test_perf:
        # (A) EVALUATION ON TEST-SPLIT OF CURRENT DATASET
        e_func.eval()
        gen_model.eval()
        dice_scores = []

        for i, data in enumerate(loader["test"], 0):
            torch.cuda.empty_cache()
            img_name, full_patch, body_patch = data
            full_patch = full_patch.to(device=device, dtype=torch.float)
                
            z_g_0 = sample_p_0(1).to(device)  
            print(z_g_0.size(), file=out_f)
            print(full_patch.size(), file=out_f)
            z_g = sample_langevin_post_z(z_g_0, full_patch, gen_model, e_func, args.po_mcmc_steps_test, args.po_mcmc_step_size, args.prior_var, args.po_mcmc_noise_var, args.gen_var, args.po_mcmc_step_gamma)
            recon_patch = gen_model.forward(z_g)
            pred_patch = torch.round(recon_patch).detach()
            dice = DiceCoeff(pred_patch, full_patch.detach())
            print(i,"/",len(loader["test"]),": Dice=", dice, file=out_f) 
            dice_scores.append(dice)
            
            # store recon
            patch_path = os.path.join(visu_path,"test_set",'patch_'+img_name[0][:-7])
            try:
                os.makedirs(patch_path)
            except OSError:
                None
            nib.save(nib.Nifti1Image(np.squeeze(pred_patch.to('cpu').numpy()[0]),affine=None),patch_path+ '/recon_patch.nii.gz')
            nib.save(nib.Nifti1Image(np.squeeze(full_patch.to('cpu').numpy()[0]),affine=None),patch_path+ '/full_patch.nii.gz')

        print(dice_scores, file=out_f)
        print(np.array(dice_scores).mean(), file=out_f)
        print(np.array(dice_scores).std(), file=out_f)
        """ 
        # (B) EVALUATION ON ADDITIONAL DATASET
        run_name = "LEBM_Vert_04_r"
        print("### Make predictions on test dataset")

        experiment_recon = 'recon4_augmheavy'
        experiment_recon_save = 'my_recon4_augmheavy'
        patients = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
        verts_recon = [1,2,3,4,5]
        testds_path = "./MRI_reconstructions/" 
        save_path = "/scratch_net/biwidl212/adolfini/vert_data/eval_test/MRI_reconstructions/"

        for patient in patients:
            print("## Patient {}".format(patient))
            for idx in range(len(verts_recon)):
                print("# Vertebra {}".format(verts_recon[idx]))
            
                hq_path  = os.path.join(testds_path, experiment_recon,patient+'_mri',str(verts_recon[idx])+'_HQ.nii.gz')
                out_path = os.path.join(save_path, experiment_recon_save,patient+'_mri',str(verts_recon[idx])+'_'+run_name +'.nii.gz')
                if not os.path.exists(hq_path):
                    print("file not found")
                    continue

                hq_nib = nib.load(hq_path)
                hq_patch = hq_nib.get_fdata().astype("int8")
                print(hq_patch.max())
                hq_patch[hq_patch!=0]=1
                print(hq_patch.min())
                print(hq_patch.max())
                hq_patch = torch.from_numpy(hq_patch).view(1,1,128,128,128).to(device=device, dtype=torch.float)

                # reconstruction
                z_g_0 = sample_p_0(1).to(device)  
                print(z_g_0.size())
                print(hq_patch.size())
                z_g = sample_langevin_post_z(z_g_0, hq_patch, gen_model, e_func, args.po_mcmc_steps_test, args.po_mcmc_step_size, args.prior_var, args.po_mcmc_noise_var, args.gen_var, args.po_mcmc_step_gamma)
                recon_patch = gen_model.forward(z_g)
                pred_patch = torch.round(recon_patch).detach() 
                nib.save(nib.Nifti1Image(np.squeeze(pred_patch.to('cpu').numpy()[0]),affine=None), out_path)

        """
