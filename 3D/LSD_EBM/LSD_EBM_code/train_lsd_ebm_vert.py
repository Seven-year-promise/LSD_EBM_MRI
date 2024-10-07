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
from model_Unet_lsdebm import E_func, VAE_new_Enc_small, VAE_new_Dec_small
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
def p_sample_progressive(noise, e_func, latent_dim, hor, e_l_steps, e_l_step_size):
    """
    Sample a sequence of latent vectors with the sequence of noise levels
    """
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

#%%Main
if  __name__ == "__main__" :   
    
    fixed_images = ['L1_verse033_seg.nii.gz', 'L1_verse265_seg.nii.gz', 'L2_verse145_seg.nii.gz',  'L3_verse082_seg.nii.gz']
    
    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')
    
    # Trianing
    parser.add_argument('--num_epochs',                         default=201, type=int, help='Number of epochs')                      
    parser.add_argument('--batch_size',                         default=4, type=int,help='Batch Size')
    parser.add_argument('--ebm_lr', type=float,                 default=0.00002,help='learning rate of the EMB network')
    #parser.add_argument('--ebm_lr', type=float,                 default=0.0001,help='learning rate of the EMB network') # like vert lebm, like classis ds
    parser.add_argument('--enc_lr', type=float,                 default=0.00002,help='learning rate of the encoder network')
    parser.add_argument('--dec_lr', type=float,                 default=0.00002,help='learning rate of the decoder network')
    parser.add_argument('--load_from_ep',                       default=None, type=int, help='checkpoint you want to load for the models') # 200
    parser.add_argument('--epoch_start',                        default=0, type=int, help='epoch you want to start from')
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
    parser.add_argument('--diff_timesteps',                     default=20, type=int, help='Number of diffusion timesteps') #30
    parser.add_argument('--beta_start',                         default=0.0001, type=float,help='Diffusion schedule')
    parser.add_argument('--beta_end',                           default=0.01, type=float,help='Diffusion schedule')

    # Testing
    parser.add_argument('--test_perf',                          default=True, help='if prediction should be performed on test dataset')
    
    args = parser.parse_args()
   
    run_name = "LSD-EBM_Vert_step_" + str(args.diff_timesteps)
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
    e_func    = E_func(args.num_latents, args.ebm_inner_dim, args.ebm_num_layers, batchnorm=args.prior_bn).to(device)
    enc_model = VAE_new_Enc_small(args.num_channels,args.num_latents).to(device) 
    dec_model = VAE_new_Dec_small(args.num_channels,args.num_latents).to(device) 


    def sample_p_0(n, sig=args.prior_var):
        return sig * torch.randn(*[n, args.num_latents])
        
    # use multiple gpus
    n_gpus=torch.cuda.device_count()
        
    #load models if requested
    if args.load_from_ep is not None:
        e_func.load_state_dict(torch.load(os.path.join(main_path,'experiments', 'efunc_'+str(args.load_from_ep)+'.pth')))
        print("LOAD EBM MODEL EP ", args.load_from_ep, file=out_f)
        enc_model.load_state_dict(torch.load(os.path.join(main_path,'experiments', 'enc_model_'+str(args.load_from_ep)+'.pth')))
        print("LOAD Encoder MODEL", args.load_from_e, file=out_fp)
        dec_model.load_state_dict(torch.load(os.path.join(main_path,'experiments', 'dec_model_'+str(args.load_from_ep)+'.pth')))
        print("LOAD Decoder MODEL", args.load_from_ep, file=out_f)
    
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

    # optimizesr
    mse = nn.MSELoss(reduction='sum')
    bce = nn.BCELoss(reduction='sum')
    if args.recon_loss=='mse':
        recon_loss = mse
    else:
        recon_loss = bce

    optE = torch.optim.Adam(e_func.parameters(),      lr=args.ebm_lr) 
    optEnc = torch.optim.Adam(enc_model.parameters(), lr=args.enc_lr) 
    optDec = torch.optim.Adam(dec_model.parameters(), lr=args.dec_lr) 
    if args.ebm_dyn_lr != None:
        schedulerE = torch.optim.lr_scheduler.ExponentialLR(optE, gamma=args.ebm_dyn_lr)
    if args.enc_dyn_lr != None:
        schedulerEnc = torch.optim.lr_scheduler.ExponentialLR(optEnc, gamma=args.enc_dyn_lr)
    if args.enc_dyn_lr != None:
        schedulerDec = torch.optim.lr_scheduler.ExponentialLR(optDec, gamma=args.dec_dyn_lr)

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
    
    # create variables as dictionaries
    dataset={}
    loader={}
    loss_e_dic={}
    loss_recon_dic={}
    loss_recon_dic_1={}
    loss_recon_dic_2={}
    loss_recon_dic_3={}
    loss_recon_dic_4={}
    loss_recon_dic_5={}
    loss_recon_dic_6={}
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
            loader[mode]=DataLoader(dataset[mode], batch_size=1, shuffle=True, drop_last=True)
        
        loss_e_dic[mode]=[]
        loss_recon_dic[mode]=[]
        loss_recon_dic_1[mode]=[]
        loss_recon_dic_2[mode]=[]
        loss_recon_dic_3[mode]=[]
        loss_recon_dic_4[mode]=[]
        loss_recon_dic_5[mode]=[]
        loss_recon_dic_6[mode]=[]
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
        epoch_v_loss_recon_1 = {}
        epoch_v_loss_recon_2 = {}
        epoch_v_loss_recon_3 = {}
        epoch_v_loss_recon_4 = {}
        epoch_v_loss_recon_5 = {}
        epoch_v_loss_recon_6 = {}
        epoch_v_dice = {}
        epoch_v_dice_proc = {}
        epoch_v_fid = {}
        
                
        for mode in modes:
            if mode != 'train' and epoch%args.ep_betw_val!=0:
                continue

            if mode=='train':
                e_func.train()
                enc_model.train()
                dec_model.train()
            else:
                e_func.eval()
                enc_model.eval()
                dec_model.eval()

            
            epoch_v_energy_real[mode]=[]
            epoch_v_energy_noise[mode]=[]
            epoch_v_energy_samp[mode]=[]
            epoch_v_loss_e[mode]=[]
            epoch_v_loss_recon[mode]=[]
            epoch_v_loss_recon_1[mode]=[]
            epoch_v_loss_recon_2[mode]=[]
            epoch_v_loss_recon_3[mode]=[]
            epoch_v_loss_recon_4[mode]=[]
            epoch_v_loss_recon_5[mode]=[]
            epoch_v_loss_recon_6[mode]=[]
            epoch_v_dice[mode]=[]
            epoch_v_dice_proc[mode]=[]
            epoch_v_fid[mode]=[]


            for i, data in enumerate(loader[mode], 0):
                torch.cuda.empty_cache()
                
                #print("mem alloc: ", torch.cuda.memory_allocated(0)/1000000)
                #print("mem max: "  , torch.cuda.get_device_properties(0).total_memory/1000000)
                
                img_name, full_patch, body_patch = data
                
                #full_patch = full_patch.to(device=device, dtype=torch.float)
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
                    optEnc.zero_grad()
                    optDec.zero_grad()
                
                
                ### START LSD-EBM ###
                ## 1. Sample x starting from x0
                z_mu, z_logvar = enc_model.forward(full_patch.to(device).float())
                z_inf = z_mu + torch.exp(0.5*z_logvar)*torch.randn_like(z_mu)

                # for how many steps to diffuse (per latent vector)
                t = np.random.randint(0, high=args.diff_timesteps, size=local_bs)
                t = torch.tensor(t, dtype=int).to(device)

                # z_pos diffused for t steps, z_neg for one additional step (noise(z_neg)>noise(z_pos))
                #z_pos, z_neg = q_sample_pairs(z_inf.detach().clone().view(-1, latent_dim), t)
                z_pos, z_neg = q_sample_pairs(z_inf.detach().clone(), t)
                # print("z_pos ", z_pos.size())
                # print("z_neg" ,z_neg.size())

                # update z_neg, try to recreate z_pos from z_neg via conditional MCMC sample
                if mode=='train':
                    z_neg_upd = sample_langevin_cond_z(z_neg.detach().clone(), t, e_func, args.cond_mcmc_steps_tr, args.cond_mcmc_step_size)
                else:
                    z_neg_upd = sample_langevin_cond_z(z_neg.detach().clone(), t, e_func, args.cond_mcmc_steps_val, args.cond_mcmc_step_size)

                # print("z_neg_upd" ,z_neg.size())

                a_s_tmp = extract(torch.tensor(a_s_prev).to(device), t + 1, z_pos.size())
                y_pos       = a_s_tmp * z_pos # adjust
                y_neg       = a_s_tmp * z_neg # adjust
                y_neg_upd   = a_s_tmp * z_neg_upd # adjust
                en_pos      = e_func(y_pos.detach().float(), t)
                en_neg_upd  = e_func(y_neg_upd.detach().float(), t)
                en_neg      = e_func(y_neg.detach().float(), t)
                #print("en pos: ", en_pos.mean())
                #print("en neg: ", en_neg.mean())

                # loss_e = - (en_pos - en_neg)
                loss_e = (en_pos - en_neg_upd) + args.EBM_reg * (en_pos**2 + en_neg_upd**2) 
                loss_scale = 1.0 / (torch.gather(torch.tensor(sigmas).to(device), 0, t + 1) / sigmas[1])
                scaled_loss_e = (loss_e*loss_scale).mean()
                loss_e = loss_e.mean()

                #recon_patch = dec_model.forward(z_inf.view(-1,latent_dim, 1, 1))
                recon_patch = dec_model.forward(z_inf)
                #print("z_inf: ", z_inf.size())
                #print("recon_patch: ", recon_patch.size())

                loss_g = (recon_loss(recon_patch, full_patch.to(device).float()) + (z_logvar + 0.5*((z_inf-z_mu)/torch.exp(z_logvar))**2).sum())/ local_bs
                # print("Loss e: ", loss_e.data)
                #print("Loss e scaled: ", scaled_loss_e.data)
                #print("Loss recon: ", loss_g.data)
                
                ## Learn
                if mode=='train':
                    ## 2.1 Learn generator and inference
                    optEnc.zero_grad()
                    optDec.zero_grad()
                    loss_g.backward()
                    optEnc.step()
                    optDec.step()

                    ## 2.2 Learn energy
                    optE.zero_grad()
                    scaled_loss_e.backward()
                    optE.step()

                del z_pos, z_neg, z_neg_upd, y_pos, y_neg, z_mu, z_logvar
                ### OLD
                ### 3. Learn prior EBM
                en_samp_v = torch.tensor([0.]) #e_func.forward(z_e.detach()) 
                en_real_v = torch.tensor([0.]) #e_func.forward(z_g.detach())
                en_noise  = en_neg.mean() #torch.tensor([0.]) #en_noise
                en_samp   = en_neg_upd.mean() #torch.tensor([0.]) #en_samp_v.mean()
                en_real   = en_pos.mean() #torch.tensor([0.]) #en_real_v.mean()
                del en_pos, en_neg, en_neg_upd

                z_gen = None
                gen_patch = None
                recon_loss_1 = torch.tensor([0])
                recon_loss_2 = torch.tensor([0])
                recon_loss_3 = torch.tensor([0])
                recon_loss_4 = torch.tensor([0])
                recon_loss_5 = torch.tensor([0])
                recon_loss_6 = torch.tensor([0])
                if mode=='validation':
                    # Generation
                    z_noise = torch.randn_like(z_inf).to(device)
                    z_gen  = p_sample_progressive(z_noise, e_func, latent_dim=args.num_latents, hor=args.diff_timesteps, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :].detach()
                    with torch.no_grad():
                        gen_patch = dec_model.forward(z_gen).detach().cpu()
                    
                    if args.save_visu and (epoch%10==0):
                        # Recon from diffusion
                        val_z_diff_1     = q_sample(z_inf, torch.ones(local_bs, dtype=int).to(device) * int(args.diff_timesteps/6)*1).detach()
                        val_x_diff_1     = dec_model(val_z_diff_1.float()).detach().cpu()
                        val_z_diff_2     = q_sample(z_inf, torch.ones(local_bs, dtype=int).to(device) * int(args.diff_timesteps/6)*2).detach()
                        val_x_diff_2     = dec_model(val_z_diff_2.float()).detach().cpu()
                        val_z_diff_3     = q_sample(z_inf, torch.ones(local_bs, dtype=int).to(device) * int(args.diff_timesteps/6)*3).detach()
                        val_x_diff_3     = dec_model(val_z_diff_3.float()).detach().cpu()
                        val_z_diff_4     = q_sample(z_inf, torch.ones(local_bs, dtype=int).to(device) * int(args.diff_timesteps/6)*4).detach()
                        val_x_diff_4     = dec_model(val_z_diff_4.float()).detach().cpu()
                        val_z_diff_5     = q_sample(z_inf, torch.ones(local_bs, dtype=int).to(device) * int(args.diff_timesteps/6)*5).detach()
                        val_x_diff_5     = dec_model(val_z_diff_5.float()).detach().cpu()
                        val_z_diff_6     = q_sample(z_inf, torch.ones(local_bs, dtype=int).to(device) * args.diff_timesteps).detach()
                        val_x_diff_6     = dec_model(val_z_diff_6.float()).detach().cpu()

                        val_z_recon_1    = p_sample_progressive(val_z_diff_1, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6), e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
                        val_x_recon_1    = dec_model(val_z_recon_1.float()).detach().cpu()
                        del val_z_recon_1, val_z_diff_1
                        val_z_recon_2    = p_sample_progressive(val_z_diff_2, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*2, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
                        val_x_recon_2    = dec_model(val_z_recon_2.float()).detach().cpu()
                        del val_z_recon_2, val_z_diff_2
                        val_z_recon_3    = p_sample_progressive(val_z_diff_3, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*3, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
                        val_x_recon_3    = dec_model(val_z_recon_3.float()).detach().cpu()
                        del val_z_recon_3, val_z_diff_3
                        val_z_recon_4    = p_sample_progressive(val_z_diff_4, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*4, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
                        val_x_recon_4    = dec_model(val_z_recon_4.float()).detach().cpu()
                        del val_z_recon_4, val_z_diff_4
                        val_z_recon_5    = p_sample_progressive(val_z_diff_5, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*5, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
                        val_x_recon_5    = dec_model(val_z_recon_5.float()).detach().cpu()
                        del val_z_recon_5, val_z_diff_5
                        val_z_recon_6    = p_sample_progressive(val_z_diff_6, e_func, latent_dim=args.num_latents, hor=args.diff_timesteps, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
                        val_x_recon_6    = dec_model(val_z_recon_6.float()).detach().cpu()
                        del val_z_recon_6, val_z_diff_6

                        recon_loss_1     = recon_loss(val_x_recon_1, full_patch.float()) 
                        recon_loss_2     = recon_loss(val_x_recon_2, full_patch.float()) 
                        recon_loss_3     = recon_loss(val_x_recon_3, full_patch.float())
                        recon_loss_4     = recon_loss(val_x_recon_4, full_patch.float())
                        recon_loss_5     = recon_loss(val_x_recon_5, full_patch.float())
                        recon_loss_6     = recon_loss(val_x_recon_6, full_patch.float())

                with torch.no_grad(): 
                    pred_patch = torch.round(recon_patch).detach() 
                    dice = DiceCoeff(pred_patch, full_patch.to(device).float()) #Note that the recon itself is not binary
                    
                    # FID score
                    dice_proc = 0.
                    fid_score = None
                    if mode=='train':
                        fid_score = 0.
                    else:
                        mu1   = z_inf.view(-1, args.num_latents).mean(dim=0).cpu()
                        mu2   = z_gen.view(-1, args.num_latents).mean(dim=0).cpu()
                        cov1  = cov(z_inf.view(-1, args.num_latents)).cpu()# z_inf.view(-1, args.num_latents).cov().cpu()
                        cov2  = cov(z_gen.view(-1, args.num_latents)).cpu()
                        fid_score = (mu1 - mu2).dot(mu1 - mu2) + np.trace(cov1 + cov2 - 2*scipy.linalg.sqrtm(cov1*cov2))

                print('[%d/%d][%d/%d]\ten_real: %.4f\ten_noise: %.4f\ten_samp: %.4f\tE_l: %.4f\tRecon_l: %.4f\tRecon_l_1: %.4f\tRecon_l_2: %.4f\tRecon_l_3: %.4f\tRecon_l_4: %.4f\tRecon_l_5: %.4f\tRecon_l_6: %.4f\tDice: %.4f\tDiceProc: %.4f\tFID: %.4f \n'% (epoch, args.num_epochs, i, len(loader[mode]), en_real.item(), en_noise.item(), en_samp.item(), loss_e.item(), loss_g.item(), recon_loss_1, recon_loss_2, recon_loss_3, recon_loss_4, recon_loss_5, recon_loss_6, dice, dice_proc, fid_score), file=out_f)
                
                epoch_v_energy_real[mode].append(en_real.item())
                epoch_v_energy_noise[mode].append(en_noise.item())
                epoch_v_energy_samp[mode].append(en_samp.item())
                epoch_v_loss_e[mode].append(loss_e.item())
                epoch_v_loss_recon[mode].append(loss_g.item())
                epoch_v_loss_recon_1[mode].append(recon_loss_1.item())
                epoch_v_loss_recon_2[mode].append(recon_loss_2.item())
                epoch_v_loss_recon_3[mode].append(recon_loss_3.item())
                epoch_v_loss_recon_4[mode].append(recon_loss_4.item())
                epoch_v_loss_recon_5[mode].append(recon_loss_5.item())
                epoch_v_loss_recon_6[mode].append(recon_loss_6.item())
                epoch_v_dice[mode].append(dice)
                epoch_v_dice_proc[mode].append(dice_proc)
                epoch_v_fid[mode].append(fid_score)
                
           
                #if args.save_visu and (epoch%20==0):
                if args.save_visu and (epoch%10==0):
                    print("STORE EXAMPLE PATCHES.", file=out_f)
                    patch_path=os.path.join(visu_path,str(epoch),mode,'patch_'+str(i)+'_'+img_name[0][:-7])
                    try:
                        os.makedirs(patch_path)
                    except OSError:
                        None
                    # save reconstruction
                    if mode=='validation' and i==0:
                        with torch.no_grad():
                            nib.save(nib.Nifti1Image(np.squeeze(pred_patch.to('cpu').numpy()[0]),affine=None), patch_path+ '/recon_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(full_patch.to('cpu').numpy()[0]),affine=None), patch_path+ '/full_patch.nii.gz')
                        del pred_patch # free memory on GPU
                    # save generation (if in validation mode)
                    # save recon from diff examples
                    if mode=='validation' and i==0:
                        gen_patch = torch.round(gen_patch)
                        with torch.no_grad():
                            nib.save(nib.Nifti1Image(np.squeeze(gen_patch.to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/gen_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_recon_1).to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/diff_recon_1_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_diff_1).to('cpu').detach().numpy()[0]),affine=None),  patch_path+ '/diff_1_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_recon_2).to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/diff_recon_2_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_diff_2).to('cpu').detach().numpy()[0]),affine=None),  patch_path+ '/diff_2_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_recon_3).to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/diff_recon_3_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_diff_3).to('cpu').detach().numpy()[0]),affine=None),  patch_path+ '/diff_3_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_recon_4).to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/diff_recon_4_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_diff_4).to('cpu').detach().numpy()[0]),affine=None),  patch_path+ '/diff_4_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_recon_5).to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/diff_recon_5_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_diff_5).to('cpu').detach().numpy()[0]),affine=None),  patch_path+ '/diff_5_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_recon_6).to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/diff_recon_6_patch.nii.gz')
                            nib.save(nib.Nifti1Image(np.squeeze(torch.round(val_x_diff_6).to('cpu').detach().numpy()[0]),affine=None),  patch_path+ '/diff_6_patch.nii.gz')
                        del gen_patch  # free memory on GPU
                    # save loss information
                    with open(os.path.join(visu_path,str(epoch),mode,"loss.txt"), "a") as f:
                        f.write("\nPatch {}: en_real: {:.6} en_noise: {:.6} en_samp: {:.6} Eloss: {:.6} MSEloss: {:.6}   VaeDice: {:.6}  FID: {:.6}".format(str(i), en_real.item(), en_noise.item(), en_samp.item(), loss_e.item(), loss_g.item(), dice, fid_score))
                    with open(os.path.join(visu_path,str(epoch),mode,"loss.csv"), "a") as f:
                        f.write("{};{};{};{};{};{};{};{}".format(str(i), en_real.item(), en_noise.item(), en_samp.item(), loss_e.item(), loss_g.item(), dice, fid_score))
                    # save stepwise generation
                    if mode=='validation' and i==0: #only for first validation batch
                        z_noise = torch.randn_like(z_inf[0,:].view(-1, args.num_latents)).to(device)
                        z_gen_steps = p_sample_progressive(z_noise, e_func, latent_dim=args.num_latents, hor=args.diff_timesteps, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size).detach()
                        with torch.no_grad():
                            for ridx in range(z_gen_steps.size()[0]):
                                x_recon_step = dec_model(z_gen_steps[ridx, :].float())
                                x_recon_step = torch.round(x_recon_step)
                                with torch.no_grad():
                                    nib.save(nib.Nifti1Image(np.squeeze(x_recon_step.to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/gen_step_'+str(ridx)+'_patch.nii.gz')
                        del z_noise, z_gen_steps
                    """
                    # save diffusion examples
                    if mode=='validation' and i==0: #only for first validation batch
                        z_diff = q_sample_progressive(z_inf[0, :].view(-1, args.num_latents).float(), args.diff_timesteps)
                        for ridx in range(z_diff.size()[0]):
                            x_diff = dec_model.forward(z_diff[ridx,:].float()).detach().cpu()
                            x_diff = torch.round(x_diff)
                            with torch.no_grad():
                                nib.save(nib.Nifti1Image(np.squeeze(x_diff.to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/diff_'+str(ridx)+'_patch.nii.gz')
                        del z_diff, x_diff
                    """
                    """
                    # recon from diff examples
                    if mode=='validation' and i==0: #only for first validation batch
                        for t_diff in [2,4,6,8,10,12]:
                            z_diff = q_sample(z_inf, torch.ones(local_bs, dtype=int).to(device) * t_diff).detach()
                            z_diff_recon = p_sample_progressive(z_noise, e_func, latent_dim=args.num_latents, hor=t_diff, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
                            with torch.no_grad():
                                x_diff_recon = dec_model(z_diff_recon[0,:].float())
                                nib.save(nib.Nifti1Image(np.squeeze(x_diff_recon.to('cpu').detach().numpy()[0]),affine=None), patch_path+ '/recon_diff_'+str(t_diff)+'_patch.nii.gz')
                    """ 
                
            loss_e_dic[mode].append(epoch_v_loss_e[mode])
            loss_recon_dic[mode].append(epoch_v_loss_recon[mode])
            loss_recon_dic_1[mode].append(epoch_v_loss_recon_1[mode])
            loss_recon_dic_2[mode].append(epoch_v_loss_recon_2[mode])
            loss_recon_dic_3[mode].append(epoch_v_loss_recon_3[mode])
            loss_recon_dic_4[mode].append(epoch_v_loss_recon_4[mode])
            loss_recon_dic_5[mode].append(epoch_v_loss_recon_5[mode])
            loss_recon_dic_6[mode].append(epoch_v_loss_recon_6[mode])
            v_dice_dic[mode].append(epoch_v_dice[mode])
            v_dice_proc_dic[mode].append(epoch_v_dice_proc[mode])
            v_fid_dic[mode].append(epoch_v_fid[mode])
            
            
            avg_v_loss_e = sum(epoch_v_loss_e[mode]) / len(epoch_v_loss_e[mode])
            avg_v_loss_recon = sum(epoch_v_loss_recon[mode]) / len(epoch_v_loss_recon[mode])
            avg_v_loss_recon_1 = sum(epoch_v_loss_recon_1[mode]) / len(epoch_v_loss_recon_1[mode])
            avg_v_loss_recon_2 = sum(epoch_v_loss_recon_2[mode]) / len(epoch_v_loss_recon_2[mode])
            avg_v_loss_recon_3 = sum(epoch_v_loss_recon_3[mode]) / len(epoch_v_loss_recon_3[mode])
            avg_v_loss_recon_4 = sum(epoch_v_loss_recon_4[mode]) / len(epoch_v_loss_recon_4[mode])
            avg_v_loss_recon_5 = sum(epoch_v_loss_recon_5[mode]) / len(epoch_v_loss_recon_5[mode])
            avg_v_loss_recon_6 = sum(epoch_v_loss_recon_6[mode]) / len(epoch_v_loss_recon_6[mode])
            avg_v_dice = sum(epoch_v_dice[mode]) / len(epoch_v_dice[mode])
            avg_v_dice_proc = sum(epoch_v_dice_proc[mode]) / len(epoch_v_dice_proc[mode])
            avg_v_fid = sum(epoch_v_fid[mode]) / len(epoch_v_fid[mode])
            avg_v_en_real = sum(epoch_v_energy_real[mode]) / len(epoch_v_energy_real[mode])
            avg_v_en_noise = sum(epoch_v_energy_noise[mode]) / len(epoch_v_energy_noise[mode])
            avg_v_en_samp = sum(epoch_v_energy_samp[mode]) / len(epoch_v_energy_samp[mode])
            
            print('\n{} Epoch: {} \t  En real: {:.4f} \t  En noise: {:.4f} \t  En samp: {:.4f} \t  ELoss: {:.4f} \t ReconLoss: {:.4f} \t ReconLoss_1: {:.4f} \t ReconLoss_2: {:.4f} \t ReconLoss_3: {:.4f} \t ReconLoss_4: {:.4f} \t ReconLoss_5: {:.4f} \t ReconLoss_6: {:.4f} \t VaeDice: {:.4} \t VaeDiceProc: {:.4} \t FID: {:.4f}%\n\n'.format(mode,epoch, avg_v_en_real, avg_v_en_noise, avg_v_en_samp, avg_v_loss_e, avg_v_loss_recon, avg_v_loss_recon_1, avg_v_loss_recon_2, avg_v_loss_recon_3, avg_v_loss_recon_4, avg_v_loss_recon_5, avg_v_loss_recon_6, avg_v_dice, avg_v_dice_proc, avg_v_fid), file=out_f)
            
            #if args.save_visu and (epoch%50==0):
            with open(os.path.join(visu_path,"epoch_loss.txt"), "a") as f:
                f.write("Epoch {}: {}  En real: {}  En noise: {}  En samp: {}  VlossE: {} VlossRecon: {} VlossRecon1: {} VlossRecon2: {} VlossRecon3: {} VlossRecon4: {} VlossRecon5: {} VlossRecon6: {} Dice:{}  DiceProc:{}  FID:{} \n".format(epoch, mode, avg_v_en_real, avg_v_en_noise, avg_v_en_samp, avg_v_loss_e, avg_v_loss_recon, avg_v_loss_recon_1, avg_v_loss_recon_2, avg_v_loss_recon_3, avg_v_loss_recon_4, avg_v_loss_recon_5, avg_v_loss_recon_6, avg_v_dice, avg_v_dice_proc, avg_v_fid))
                if mode == 'validation':
                    print("\n", file=out_f)
            with open(os.path.join(visu_path,"epoch_loss.csv"), "a") as f:
                f.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{} \n".format(epoch, mode, avg_v_en_real, avg_v_en_noise, avg_v_en_samp, avg_v_loss_e, avg_v_loss_recon, avg_v_loss_recon_1, avg_v_loss_recon_2, avg_v_loss_recon_3, avg_v_loss_recon_4, avg_v_loss_recon_5, avg_v_loss_recon_6, avg_v_dice, avg_v_dice_proc, avg_v_fid))
            
        #Saving current model
        if epoch%20==0:
            model_path = os.path.join(main_path,'experiments')
            if not os.path.exists(model_path):
                print('Creating directory at:', model_path, file=out_f)
                os.makedirs(model_path)
            torch.save(e_func.state_dict(),os.path.join(model_path,'efunc_'+str(epoch)+'.pth'))
            torch.save(enc_model.state_dict(),os.path.join(model_path,'enc_model_'+str(epoch)+'.pth'))
            torch.save(dec_model.state_dict(),os.path.join(model_path,'dec_model_'+str(epoch)+'.pth'))
            with open(os.path.join(model_path,args.experiment+str(epoch)+".p"), 'wb') as f:
                pickle.dump([loss_e_dic, loss_recon_dic, v_dice_dic, v_dice_proc_dic, v_fid_dic],f)
        
        if args.ebm_dyn_lr != None:
            schedulerE.step()
        if args.enc_dyn_lr != None:
            schedulerEnc.step()
        if args.dec_dyn_lr != None:
            schedulerDec.step()
       
        print("--- %s seconds ---" % (time.time() - start_time), file=out_f)

    # TEST PERFORMANCE
    if args.test_perf:
        e_func.eval()
        enc_model.eval()
        dec_model.eval()


        """
        # (A) EVALUATION ON VALIDATION-SPLIT FOR DIF DIFF LEVELS
        dice_scores = []
        dice_scores_1 = []
        dice_scores_2 = []
        dice_scores_3 = []
        dice_scores_4 = []
        dice_scores_5 = []
        dice_scores_6 = []
        # BS OF VALIDATION SET MUST BE SET TO 1
        for i, data in enumerate(loader["validation"], 0):
            print("ON VALIDATION SET")

            torch.cuda.empty_cache()
            img_name, full_patch, body_patch = data
               
            z_mu, z_logvar = enc_model.forward(full_patch.to(device).float())
            z_inf = z_mu + torch.exp(0.5*z_logvar)*torch.randn_like(z_mu)
            
            recon_patch = dec_model.forward(z_inf)
            pred_patch = torch.round(recon_patch).detach()
            dice = DiceCoeff(pred_patch, full_patch.detach().to(device).float())
            dice_scores.append(dice)

            # Recon from diffusion
            val_z_diff_1     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * int(args.diff_timesteps/6)*1).detach()
            val_z_diff_2     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * int(args.diff_timesteps/6)*2).detach()
            val_z_diff_3     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * int(args.diff_timesteps/6)*3).detach()
            val_z_diff_4     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * int(args.diff_timesteps/6)*4).detach()
            val_z_diff_5     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * int(args.diff_timesteps/6)*5).detach()
            val_z_diff_6     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * int(args.diff_timesteps/6)*6).detach()

            val_z_recon_1    = p_sample_progressive(val_z_diff_1, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*1, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
            val_x_recon_1    = torch.round(dec_model(val_z_recon_1.float()).detach())
            dice1 = DiceCoeff(val_x_recon_1, full_patch.detach().to(device).float())
            dice_scores_1.append(dice1)
            del val_z_recon_1, val_z_diff_1, val_x_recon_1
            val_z_recon_2    = p_sample_progressive(val_z_diff_2, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*2, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
            val_x_recon_2    = torch.round(dec_model(val_z_recon_2.float()).detach())
            dice2 = DiceCoeff(val_x_recon_2, full_patch.detach().to(device).float())
            dice_scores_2.append(dice2)
            del val_z_recon_2, val_z_diff_2, val_x_recon_2
            val_z_recon_3    = p_sample_progressive(val_z_diff_3, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*3, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
            val_x_recon_3    = torch.round(dec_model(val_z_recon_3.float()).detach())
            dice3 = DiceCoeff(val_x_recon_3, full_patch.detach().to(device).float())
            dice_scores_3.append(dice3)
            del val_z_recon_3, val_z_diff_3, val_x_recon_3
            val_z_recon_4    = p_sample_progressive(val_z_diff_4, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*4, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
            val_x_recon_4    = torch.round(dec_model(val_z_recon_4.float()).detach())
            dice4 = DiceCoeff(val_x_recon_4, full_patch.detach().to(device).float())
            dice_scores_4.append(dice4)
            del val_z_recon_4, val_z_diff_4, val_x_recon_4
            val_z_recon_5    = p_sample_progressive(val_z_diff_5, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*5, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
            val_x_recon_5    = torch.round(dec_model(val_z_recon_5.float()).detach())
            dice5 = DiceCoeff(val_x_recon_5, full_patch.detach().to(device).float())
            dice_scores_5.append(dice5)
            del val_z_recon_5, val_z_diff_5, val_x_recon_5
            val_z_recon_6    = p_sample_progressive(val_z_diff_6, e_func, latent_dim=args.num_latents, hor=int(args.diff_timesteps/6)*6, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
            val_x_recon_6    = torch.round(dec_model(val_z_recon_6.float()).detach())
            dice6 = DiceCoeff(val_x_recon_6, full_patch.detach().to(device).float())
            dice_scores_6.append(dice6)
            del val_z_recon_6, val_z_diff_6, val_x_recon_6

            print(i,"/",len(loader["test"]),": Dice=", dice) 
            print(i,"/",len(loader["test"]),": Dices=", dice1, ", ", dice2, ", ", dice3, ", ", dice4, ", ", dice5, ", ", dice6) 

        print(dice_scores)
        print("dice")
        print(np.array(dice_scores).mean())
        print(np.array(dice_scores).std())
        print("dice 1")
        print(np.array(dice_scores_1).mean())
        print(np.array(dice_scores_1).std())
        print("dice 2")
        print(np.array(dice_scores_2).mean())
        print(np.array(dice_scores_2).std())
        print("dice 3")
        print(np.array(dice_scores_3).mean())
        print(np.array(dice_scores_3).std())
        print("dice 4")
        print(np.array(dice_scores_4).mean())
        print(np.array(dice_scores_4).std())
        print("dice 5")
        print(np.array(dice_scores_5).mean())
        print(np.array(dice_scores_5).std())
        print("dice 6")
        print(np.array(dice_scores_6).mean())
        print(np.array(dice_scores_6).std())
        """
        
        # (B) EVALUATION ON TEST-SPLIT
        dice_scores = []
        for i, data in enumerate(loader["test"], 0):

            torch.cuda.empty_cache()
            img_name, full_patch, body_patch = data
               
            z_mu, z_logvar = enc_model.forward(full_patch.to(device).float())
            z_inf = z_mu + torch.exp(0.5*z_logvar)*torch.randn_like(z_mu)
            
            # diffusion
            t_diff = int(args.diff_timesteps/6)*2 # set number of diff. steps to perform 
            val_z_diff     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * t_diff).detach()
            val_z_recon    = p_sample_progressive(val_z_diff, e_func, latent_dim=args.num_latents, hor=t_diff, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
            
            recon_patch = dec_model.forward(val_z_recon.float())
            pred_patch = torch.round(recon_patch).detach()
            dice = DiceCoeff(pred_patch, full_patch.detach().to(device).float())
            dice_scores.append(dice)
            print(i,"/",len(loader["test"]),": Dice=", dice, file=out_f) 
            
            # store recon
            patch_path = os.path.join(visu_path,"test_set",'patch_'+img_name[0][:-7])
            try:
                os.makedirs(patch_path)
            except OSError:
                None
            nib.save(nib.Nifti1Image(np.squeeze(pred_patch.to('cpu').numpy()[0]),affine=None),patch_path+ '/recon_patch.nii.gz')
            nib.save(nib.Nifti1Image(np.squeeze(full_patch.to('cpu').numpy()[0]),affine=None),patch_path+ '/full_patch.nii.gz')

        print(dice_scores, file=out_f)
        print("dice", file=out_f)
        print(np.array(dice_scores).mean(), file=out_f)
        print(np.array(dice_scores).std(), file=out_f)

        # (C) EVALUATION ON ADDITIONAL DATASET
        #run_name = "lebmXXX_base_testGenVar05"
        print("### Make predictions on test dataset")

        experiment_recon = 'recon4_augmheavy'
        experiment_recon_save = 'my_recon4_augmheavy'
        patients = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
        verts_recon = [1,2,3,4,5]
        testds_path = "./../MRI_reconstructions/" 
        save_path = "./../Vert_data/eval_test_step_"+ str(args.diff_timesteps)+ "/MRI_reconstructions/"

        for patient in patients:
            print("## Patient {}".format(patient), file=out_f)
            for idx in range(len(verts_recon)):
                print("# Vertebra {}".format(verts_recon[idx]), file=out_f)
            
                hq_path  = os.path.join(testds_path, experiment_recon,patient+'_mri',str(verts_recon[idx])+'_HQ.nii.gz')
                out_path = os.path.join(save_path, experiment_recon_save,patient+'_mri')
                if not os.path.exists(out_path):
                    print('Creating directory at:', out_path, file=out_f)
                    os.makedirs(out_path)
                out_path = os.path.join(out_path, str(verts_recon[idx])+'_'+run_name +'.nii.gz')
                
                if not os.path.exists(hq_path):
                    print("file not found", file=out_f)
                    continue

                hq_nib = nib.load(hq_path)
                hq_patch = hq_nib.get_fdata().astype("int8")
                print(hq_patch.max(), file=out_f)
                hq_patch[hq_patch!=0]=1
                print(hq_patch.min(), file=out_f)
                print(hq_patch.max(), file=out_f)
                full_patch = torch.from_numpy(hq_patch).view(1,1,128,128,128).to(device=device, dtype=torch.float)

                z_mu, z_logvar = enc_model.forward(full_patch.to(device).float())
                z_inf = z_mu + torch.exp(0.5*z_logvar)*torch.randn_like(z_mu)
            
                # diffusion
                t_diff = int(args.diff_timesteps/6)*2 # set number of diff. steps to perform 
                val_z_diff     = q_sample(z_inf, torch.ones(1, dtype=int).to(device) * t_diff).detach()
                val_z_recon    = p_sample_progressive(val_z_diff, e_func, latent_dim=args.num_latents, hor=t_diff, e_l_steps=args.cond_mcmc_steps_val, e_l_step_size=args.cond_mcmc_step_size)[0, :, :]
            
                recon_patch = dec_model.forward(val_z_recon.float())
                pred_patch = torch.round(recon_patch).detach() 
                nib.save(nib.Nifti1Image(np.squeeze(pred_patch.to('cpu').numpy()[0]),affine=None), out_path)

    out_f.close()

