### Imports
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import numpy as np
from numpy import arange as nparange
from numpy import sqrt as npsqrt
from numpy import random as nprandom
from numpy import rollaxis as nprollaxis
from scipy.linalg import sqrtm as scipysqrtm
from numpy import trace as nptrace
from fid_score import fid_from_samples, m_s_from_samples

### Set running mode to CPU or GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device set to: ", device)

### Parameters ###
seed_all = 759948# 798234
dataset = "FashionMNIST"     # MNIST, CIFAR10, CelebA64, CelebA128

#main_path = "./"        # Target dir for all outputs (model-files, tsb-files, ...)
main_path = "./FashionMNIST/" 
DEBUG = False           # If debug information should be printed

latent_dim  = 100       # Size of latent-vectors
m           = 128       # Batch-size
epochs      = 200       # Number of epochs to train model
use_buffer  = False     # If a sample-buffer should be used for sampling
loss_reg    = 0.        # Regularizing term in loss (0 = No regularization)
batchnorm   = True      # If batch normalization is applied
live_plotting = False   # If plots should be created live (e.g. when running as notebook)
tsb_tracking  = True    # Track training process and store to tsb file
store_every   = 10      # Store model every ... epochs (None = never)
eval_every    = 2       # Validation evaluation every ... epochs (None = never)
mcmc_steps    = 30      # Number of langevin mcmc steps 
mcmc_step_size = 0.01   # Langevin mcmc step size 
diff_timesteps = 12     # Number of diffusion timesteps 
beta_start=0.0001       # diffusion schedule 
beta_end=0.01           # diffusion schedule 

betas = np.linspace(beta_start, beta_end, 1000)
betas = np.append(betas, 1.)
sqrt_alphas = np.sqrt(1. - betas)
idx = np.concatenate([np.arange(diff_timesteps) * (1000 // ((diff_timesteps - 1) * 2)), [999]])
a_s = np.concatenate(
    [[np.prod(sqrt_alphas[: idx[0] + 1])],
     np.asarray([np.prod(sqrt_alphas[idx[i - 1] + 1: idx[i] + 1]) for i in np.arange(1, len(idx))])])
sigmas = np.sqrt(1 - a_s ** 2)
a_s_cum = np.cumprod(a_s)
sigmas_cum = np.sqrt(1 - a_s_cum ** 2)
a_s_prev = a_s.copy()
a_s_prev[-1] = 1

# Print to console
print("### Parameter choice ###")
print(f"""dataset       = {dataset}, (MNIST, CIFAR10, CelebA64, CelebA128)""")
print(f"""seed_all      = {seed_all}""")
print(f"""main_path     = {main_path}, Target dir for all outputs (model-files, tsb-files, ...)""")
print(f"""DEBUG         = {DEBUG} ,If debug information should be printed""")
print(f"""latent_dim    = {latent_dim}, Size of latent-vectors""")
print(f"""m             = {m}, Batch-size""")
print(f"""epochs        = {epochs}, Number of epochs to train model""")
print(f"""use_buffer    = {use_buffer}, If a sample-buffer should be used for sampling""")
print(f"""loss_reg      = {loss_reg}, Regularizing term in loss (0 = No regularization)""")
print(f"""batchnorm     = {batchnorm}, If batch normalization is applied""")
print(f"""live_plotting = {live_plotting}, If plots should be created live (e.g. when running as notebook)""")
print(f"""tsb_tracking  = {tsb_tracking}, Track training process and store to tsb file""")
print(f"""store_every   = {store_every}, Store model every ... epochs (None = never)""")
print(f"""eval_every    = {eval_every}, Validation evaluation every ... epochs (None = never)""")
print(f"""mcmc steps    = {mcmc_steps}, Number of langevin mcmc steps""")
print(f"""mcmc step size = {mcmc_step_size}, Langevin mcmc step size""")
print(f"""diff timesteps= {diff_timesteps}, Number of diffusion timesteps""")
print(f"""beta start    = {beta_start}, Diffusion schedule""")
print(f"""beta end      = {beta_end}, Diffusion schedule""")
print("########################")
##################

### Seeding
"""
Remark on reproducibility:
"Completely reproducible results are not guaranteed across PyTorch releases, individual commits,
    or different platforms. Furthermore, results may not be reproducible between CPU and GPU
    executions, even when using identical seeds."
"""
torch.manual_seed(seed_all)
random.seed(seed_all) # py
nprandom.seed(seed_all)
init_fid = True

### Import dataset
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    nprandom.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed_all)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.), (1.))])

if dataset == "FashionMNIST":
    print("Load FashionMNIST dataset.")
    train_dataset = torchvision.datasets.FashionMNIST(root=main_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=main_path,
                                          train=False,
                                          transform=transform)
    channels = 1
    padding  = 4
elif dataset == "CIFAR10":
    print("Load CIFAR10 dataset.")
    train_dataset = torchvision.datasets.CIFAR10(root=main_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=main_path,
                                          train=False,
                                          transform=transform)
    channels = 3
    padding  = 0
elif dataset == "CelebA64":
    print("Load CelebA dataset.")
    crop = lambda x: transforms.functional.crop(x, 45, 25, 173-45, 153-25)
    train_dataset = torchvision.datasets.CelebA(root=main_path, split='train', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(64),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.), (1.))]))
    test_dataset = torchvision.datasets.CelebA(root=main_path, split='valid', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(64),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.), (1.))]))
    channels = 3
    padding = -64
elif dataset == "CelebA128":
    print("Load CelebA dataset.")
    crop = lambda x: transforms.functional.crop(x, 45, 25, 173-45, 153-25)
    train_dataset = torchvision.datasets.CelebA(root=main_path, split='train', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(128),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.), (1.))]))
    test_dataset = torchvision.datasets.CelebA(root=main_path, split='valid', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(128),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.), (1.))]))
    channels = 3
    padding = -128
else:
    print("ERROR: Dataset not found.")
    exit()

img_shape = train_dataset[0][0].size()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=m,
                                           shuffle=True,
                                           worker_init_fn=seed_worker,
                                           generator=g,)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=m,
                                          shuffle=False,
                                          worker_init_fn=seed_worker,
                                          generator=g,)

"""
x1 = nprollaxis(iter(train_loader).next()[0].numpy(), 1, 4)*255
print("*** ", x1.shape)
x2 = nprollaxis(iter(test_loader).next()[0].numpy(), 1, 4)*255
print("*** ", x2.shape)
print("fid test: ", fid_from_samples(x1, x2))
"""

### Model definition - EBM prior
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.gelu(x)

# Input  = batch of latent vectors (m x latent_dim)
# Output = energy value per latent vecotr (m)
class E_func(nn.Module):
    def __init__(self, latent_dim=latent_dim, inner_dim=1000, out_dim=1):
        super(E_func, self).__init__()
        # prior layers
        self.out_dim = out_dim

        apply_sn = lambda x: x
        f = GELU()

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(latent_dim+1, inner_dim)),
            f,
            apply_sn(nn.Linear(inner_dim, inner_dim)),
            f,
            apply_sn(nn.Linear(inner_dim, out_dim))
        )

    def forward(self, z, t):
        zt = torch.cat([z, t.view(-1,1)], axis=1)
        e = self.ebm(zt)
        return  e.view(-1, self.out_dim, 1, 1)

### Test prior model
if DEBUG:
    # ... with CPU
    E_fun_TEST= E_func(latent_dim)
    z_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=True)
    en_TEST = E_fun_TEST.forward(z_TEST, torch.tensor([2,3]))
    print("Prior model CPU test: Passing latent vectors of size ", z_TEST.size(), " to energy-network resulted in output of size ", en_TEST.size())

    # ... with GPU
    E_fun_TEST= E_func(latent_dim).to(device)
    z_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=True).to(device)
    en_TEST = E_fun_TEST.forward(z_TEST, torch.tensor([2,3]).to(device))
    print("Prior model GPU test: Passing latent vectors of size ", z_TEST.size(), " to energy-network resulted in output of size ", en_TEST.size())

### Model definition - Generation model
g_llhd_sigma = 0.3
mseloss = nn.MSELoss(reduction='sum')
class GenModel(nn.Module):
    def __init__(self, latent_dim=latent_dim, inner_dim=128, num_chan=channels, img_len=img_shape[1]):
        super(GenModel, self).__init__()
        g_batchnorm = batchnorm
        f = nn.LeakyReLU(0.2)
        self.gen = None
        if img_len <= 38: # for images up to size 38x38
            self.gen = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, inner_dim*16, 4, 1, 0, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*16) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*16, inner_dim*8, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*8) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*8, inner_dim*4, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*4) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*4, inner_dim*2, 4, 1, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*2) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*2, inner_dim*1, 4, 1, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*1) if g_batchnorm else nn.Identity(),
                f,
                # nn.ConvTranspose2d(inner_dim*1, num_chan, 4, 2, 1),
                nn.ConvTranspose2d(inner_dim*1, num_chan, 4, 2, int(0.5*(38-img_len))),
                nn.Tanh()
                )
        elif img_len == 64:
            self.gen = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, inner_dim*16, 4, 1, 0, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*16) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*16, inner_dim*8, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*8) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*8, inner_dim*4, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*4) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*4, inner_dim*2, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*2) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*2, inner_dim*1, 4, 1, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*1) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*1, num_chan, 4, 2, 2),
                nn.Tanh()
            )
        elif img_len == 128:
            self.gen = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, inner_dim*16, 4, 1, 0, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*16) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*16, inner_dim*8, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*8) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*8, inner_dim*4, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*4) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*4, inner_dim*2, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*2) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*2, inner_dim*1, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*1) if g_batchnorm else nn.Identity(),
                f,
                nn.ConvTranspose2d(inner_dim*1, num_chan, 4, 2, 1),
                nn.Tanh()
            )

    def forward(self, z):
        x = 0.5*self.gen(z)+0.5
        return x

### Test generation model
if DEBUG:
    # ... with CPU
    model_gen_TEST= GenModel(latent_dim)
    z_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=True)
    z_TEST = z_TEST.view(-1, latent_dim, 1, 1)
    x_of_z_TEST = model_gen_TEST.forward(z_TEST)
    print("Generation model CPU test: Passing latent vectors of size ", z_TEST.size(), " to generation-network resulted in output of size ", x_of_z_TEST.size())

    # ... with GPU
    model_gen_TEST= GenModel(latent_dim).to(device)
    z_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=True)
    z_TEST = z_TEST.view(-1, latent_dim, 1, 1).to(device)
    x_of_z_TEST = model_gen_TEST.forward(z_TEST)
    print("Generation model GPU test: Passing latent vectors of size ", z_TEST.size(), " to generation-network resulted in output of size ", x_of_z_TEST.size())


### Model definition - Inference model
class InfModel(nn.Module):
    def __init__(self, latent_dim=latent_dim, inner_dim=128, num_chan=channels, img_len=img_shape[1]):
        super(InfModel, self).__init__()
        g_batchnorm = batchnorm
        f = nn.LeakyReLU(0.2)
        self.inf = None
        if img_len <= 38: # for images up to size 38x38
            self.inf = nn.Sequential(
                nn.Conv2d(num_chan, inner_dim, 4, 1, 0, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*1) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim, inner_dim*2, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*2) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim*2, inner_dim*4, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*4) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim*4, inner_dim*8, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*8) if g_batchnorm else nn.Identity(),
                f,
                # dense
                nn.Flatten(),
                nn.Linear(inner_dim*9*8, inner_dim*8),
                f,
                nn.Linear(inner_dim*8, inner_dim*4),
                f,
                nn.Linear(inner_dim*4, inner_dim),
                f,
                )
        elif img_len == 64:
            self.inf = nn.Sequential(
                nn.Conv2d(num_chan, inner_dim, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*1) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim, inner_dim*2, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*2) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim*2, inner_dim*4, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*4) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim*4, inner_dim*8, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*8) if g_batchnorm else nn.Identity(),
                f,
                # dense
                nn.Flatten(),
                nn.Linear(inner_dim*8*4*4, inner_dim*8),
                f,
                nn.Linear(inner_dim*8, inner_dim*4),
                f,
                nn.Linear(inner_dim*4, inner_dim),
                f,
            )
        elif img_len == 128:
            self.inf = nn.Sequential(
                nn.Conv2d(num_chan, inner_dim, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*1) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim, inner_dim*2, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*2) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim*2, inner_dim*4, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*4) if g_batchnorm else nn.Identity(),
                f,
                nn.Conv2d(inner_dim*4, inner_dim*8, 4, 2, 1, bias = not g_batchnorm),
                nn.BatchNorm2d(inner_dim*8) if g_batchnorm else nn.Identity(),
                f,
                # dense
                nn.Flatten(),
                nn.Linear(inner_dim*8*8*8, inner_dim*8),
                f,
                nn.Linear(inner_dim*8, inner_dim*4),
                f,
                nn.Linear(inner_dim*4, inner_dim),
                f,
            )
        self.mu     = nn.Sequential(nn.Linear(inner_dim, latent_dim))
        self.logvar = nn.Sequential(nn.Linear(inner_dim, latent_dim))

    def forward(self, z):
        z_pre = self.inf(z)
        mu, logvar = self.mu(z_pre), self.logvar(z_pre)
        z = mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
        return z, mu, logvar

### Test generation model
if DEBUG:
    # ... with CPU
    model_inf_TEST= InfModel(latent_dim)
    x_TEST = next(iter(train_loader))[0][:2, : ,:, :]
    print(x_TEST.size())
    z_of_x_TEST = model_inf_TEST.forward(x_TEST)
    print("Generation model CPU test: Passing latent vectors of size ", x_TEST.size(), " to generation-network resulted in output of size ", z_of_x_TEST.size())

    # ... with GPU
    model_inf_TEST= InfModel(latent_dim).to(device)
    x_TEST = next(iter(train_loader))[0][:2, : ,:, :].to(device)
    z_of_x_TEST = model_inf_TEST.forward(x_TEST)
    print("Generation model GPU test: Passing latent vectors of size ", x_TEST.size(), " to generation-network resulted in output of size ", z_of_x_TEST.size())


### Diffusion functions and helper
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
def q_sample_progressive(z_0):
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

# Test diffusion of latent vecotr: q_sample() & q_sample_progressive()
if DEBUG:
  ## CPU
  print("TEST CPU")
  z_0_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=False)

  inf_TEST = InfModel(latent_dim)
  z_0_TEST = inf_TEST.forward(next(iter(train_loader))[0][:2, : ,:, :])

  # test q_sample()
  t_now = torch.ones(z_0_TEST.size()[0], dtype=int)*3
  z_TEST = q_sample(z_0_TEST.view(-1, latent_dim), t_now)
  print("z_0_TEST ", z_0_TEST.size())
  print("z_TEST " ,z_TEST.size())
  # test q_sample_progressive()
  z_TEST = q_sample_progressive(z_0_TEST.view(-1, latent_dim))
  print("z_0_TEST ", z_0_TEST.size())
  print("z_TEST ", z_TEST.size())
  print(z_TEST.device)

  if live_plotting:
    gen_TEST = GenModel(latent_dim)
    x_TEST = gen_TEST.forward(z_TEST[:,0,:].view(-1, latent_dim, 1, 1).float())
    print("x_TEST ", x_TEST.size())
    for it in range(7):
      plt.subplot(1, 7, it+1)
      plt.imshow(np.moveaxis(x_TEST[it, :, :, :].detach().numpy(), 0, -1))
    plt.show()

  ## GPU
  print("TEST GPU")
  z_0_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=False).to(device)
  # test q_sample()
  t_now = (torch.ones(z_0_TEST.size()[0], dtype=int)*3).to(device)
  z_TEST = q_sample(z_0_TEST, t_now).to(device)
  print(z_0_TEST.size())
  print(z_TEST.size())
  # test q_sample_progressive()
  z_TEST = q_sample_progressive(z_0_TEST).to(device)
  print(z_0_TEST.size())
  print(z_TEST.size())
  print(z_TEST.device)


def sample_langevin_cond_z(z_tilde, t, netE, e_l_steps=mcmc_steps, e_l_step_size=mcmc_step_size, e_prior_sig=1., noise_sig=0.01, e_l_with_noise = True, verbose=False):
        sigma = extract(torch.tensor(sigmas).to(z_tilde.device), t + 1, z_tilde.size())
        sigma_cum = extract(torch.tensor(sigmas_cum).to(z_tilde.device), t, z_tilde.size())
        a_s = extract(torch.tensor(a_s_prev).to(z_tilde.device), t + 1, z_tilde.size())

        netE.eval()
        for p in netE.parameters():
            p.requires_grad = False
        # TODO keep like this?
        # y = z_tilde.clone().detach()
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
                y.data.add_(npsqrt(e_l_step_size) * noise.data)

            y.grad.detach_()
            y.grad.zero_()

        for p in netE.parameters():
            p.requires_grad = True
        netE.train()
        z = (y / a_s).float()
        return z.detach()

### Test Langevin MCMC sampling - Prior
if DEBUG:
    # ... with CPU
    E_fun_TEST= E_func(latent_dim)

    # z_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=True)

    inf_TEST = InfModel(latent_dim)
    x_TEST = next(iter(train_loader))[0][:2, : ,:, :]
    z_TEST = inf_TEST.forward(x_TEST).detach().view(-1, latent_dim)

    y_0_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=True)
    t_TEST = torch.tensor([2,3], dtype=int)
    print("*")

    z_sample_TEST = sample_langevin_cond_z(z_TEST, t_TEST, E_fun_TEST)
    print("*")
    print("Langevin sampling of Prior, CPU test: Sampling starting from initial data of size ", z_TEST.size(), \
          " with energies ", E_fun_TEST(z_TEST, t_TEST).detach().numpy().squeeze(), " resultet in samples of size ", z_sample_TEST.size(), \
          " with energies ", E_fun_TEST(z_sample_TEST.view(-1, latent_dim), t_TEST).detach().numpy().squeeze(), ".")

    if live_plotting:
        gen_TEST = GenModel(latent_dim)
        x_sample_TEST = gen_TEST.forward(z_TEST.view(-1, latent_dim, 1, 1).float())
        print("x_TEST ", x_TEST.size())
        for it in range(2):
            plt.subplot(2, 2, it*2+1)
            plt.imshow(np.rollaxis(x_TEST[it, :, : ,:].detach().numpy(), 0, 3))
            plt.subplot(2, 2, it*2+2)
            plt.imshow(np.rollaxis(x_sample_TEST[it,: ,: ,:].detach().numpy(), 0, 3))
        plt.show()

    # ... with GPU
    E_fun_TEST= E_func(latent_dim).to(device)
    z_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=True).to(device)
    y_0_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=True).to(device)
    t_TEST = torch.tensor([2,3], dtype=int).to(device)
    z_sample_TEST = sample_langevin_cond_z(z_TEST, t_TEST, E_fun_TEST)

    print("Langevin sampling of prior, GPU test: Sampling starting from initial data of size ", z_TEST.size(), \
          " with energies ", E_fun_TEST(z_TEST, t_TEST).cpu().detach().numpy().squeeze(), " resultet in samples of size ", z_sample_TEST.size(), \
          " with energies ", E_fun_TEST(z_sample_TEST, t_TEST).cpu().detach().numpy().squeeze(), ".")

# Starting from noise latent vector (b x latent_dim) 
#   returns (diff_steps x b x latent_dim) with noise vectors last and reduced noise via MCMC sampling towards beginning
def p_sample_progressive(noise, e_func, hor=diff_timesteps):
    """
    Sample a sequence of latent vectors with the sequence of noise levels
    """
    num = noise.shape[0]
    z_neg_t = noise # b x latent_dim
    #z_neg = torch.zeros((diff_timesteps,) + noise.size()).to(noise.device) # diff_timesteps x b x latent_dim
    z_neg = torch.zeros((hor,) + noise.size()).to(noise.device) # diff_timesteps x b x latent_dim
    z_neg = torch.cat([z_neg, noise.view(1, num, -1)], axis=0) # (diff_timesteps+1) x b x latent_dim
    for t in range(hor - 1, -1, -1):
      # print("pre ", e_func.forward(z_neg_t, torch.tensor([t]*num).to(noise.device)).detach().numpy().flatten())
      z_neg_t = sample_langevin_cond_z(z_neg_t, torch.tensor([t]*num).to(noise.device), e_func) # b x latent_dim
      z_neg_t = z_neg_t.view(num, latent_dim) # useless?
      # print("post ", e_func.forward(z_neg_t, torch.tensor([t]*num).to(noise.device)).detach().numpy().flatten())
      #insert_mask = (torch.ones(hor+1)*t == torch.range(0,diff_timesteps)).float().to(noise.device)
      insert_mask = (torch.ones(hor+1)*t == torch.range(0, hor)).float().to(noise.device)
      #insert_mask = (torch.ones(diff_timesteps+1)*t == torch.range(0,diff_timesteps)).float().to(noise.device)
      insert_mask = torch.stack([torch.stack([insert_mask]*noise.size()[1], axis=1)]*noise.size()[0], axis=1) # latent_timesteps x b x latent_dim
      # print(torch.stack([z_neg_t]*(diff_timesteps+1)).size())
      #z_neg = insert_mask * torch.stack([z_neg_t]*(diff_timesteps+1)) + (1. - insert_mask) * z_neg
      z_neg = insert_mask * torch.stack([z_neg_t]*(hor+1)) + (1. - insert_mask) * z_neg
    return z_neg

# Test latent vecotor from noise: p_sample_progressive()
if DEBUG:
  ## CPU
  print("TEST CPU")
  E_fun_TEST= E_func(latent_dim)
  noise_TEST = torch.randn((16,latent_dim) )
  tmp = p_sample_progressive(noise_TEST, E_fun_TEST)
  print(tmp.size())
  print(tmp.device)

  if live_plotting:
    gen_TEST = GenModel(latent_dim)
    x_sample_TEST = gen_TEST.forward(tmp[:, 0, :].view(-1, latent_dim, 1, 1).float())
    for it in range(7):
      plt.subplot(1, 7, it+1)
      plt.imshow(np.rollaxis(x_sample_TEST[it,: ,: ,:].detach().numpy(), 0, 3))
    plt.show()
  
  ## GPU
  print("TEST GPU")
  E_fun_TEST= E_func(latent_dim).to(device)
  noise_TEST = torch.randn((16,latent_dim)).to(device)
  tmp = p_sample_progressive(noise_TEST, E_fun_TEST)
  print(tmp.size())
  print(tmp.device)

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

# test q_sample_pairs
if DEBUG:
  ## CPU
  print("TEST CPU")

  # z_0_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=False)
  
  inf_TEST = InfModel(latent_dim)
  x_TEST = next(iter(train_loader))[0][:2, : ,:, :]
  z_0_TEST = inf_TEST.forward(x_TEST).detach().view(-1, latent_dim)
  
  print(z_0_TEST.size())
  z_TEST, z_TEST_p1 = q_sample_pairs(z_0_TEST, torch.tensor([2,3], dtype=int))
  print(z_TEST.size())
  print(z_TEST.device)
  print(z_TEST_p1.size())
  print(z_TEST_p1.device)

  if live_plottting:
    gen_TEST = GenModel(latent_dim)
    x_samp_TEST = gen_TEST.forward(z_TEST.view(-1, latent_dim, 1, 1).float())
    x_p1_samp_TEST = gen_TEST.forward(z_TEST_p1.view(-1, latent_dim, 1, 1).float())
    for it in range(2):
      plt.subplot(2, 3, 1+it*3)
      plt.imshow(np.rollaxis(x_TEST[it,: ,: ,:].detach().numpy(), 0, 3))
      plt.subplot(2, 3, 2+it*3)
      plt.imshow(np.rollaxis(x_samp_TEST[it,: ,: ,:].detach().numpy(), 0, 3))
      plt.subplot(2, 3, 3+it*3)
      plt.imshow(np.rollaxis(x_p1_samp_TEST[it,: ,: ,:].detach().numpy(), 0, 3))
    plt.show()

  ## CPU
  print("TEST GPU")
  z_0_TEST = torch.tensor([[0.]*latent_dim, [1.]*latent_dim], requires_grad=False).to(device)
  print(z_0_TEST.size())
  z_TEST, z_TEST_p1 = q_sample_pairs(z_0_TEST, torch.tensor([2,3], dtype=int).to(device))
  print(z_TEST.size())
  print(z_TEST.device)
  print(z_TEST_p1.size())
  print(z_TEST_p1.device)


### Sampling batch from random normal P0
e_init_sig = 1.
def sample_p_0(n, sig=e_init_sig):
        return sig * torch.randn([n, latent_dim, 1, 1])

### Helpers for loading and saveing models
def save_model(model, path):
    torch.save(model.state_dict(), path)
def load_model(model, path):
    model.load_state_dict(torch.load(path))


### Set up model and optimizer
e_func = E_func(latent_dim).to(device)
gen_model = GenModel(latent_dim).to(device)
inf_model = InfModel(latent_dim).to(device)
pre_run_epochs = 200

# LOAD PREVIOUS STATS OF MODEL
load_model(e_func,    main_path + "e_func_ep"    + str(pre_run_epochs))
load_model(gen_model, main_path + "gen_model_ep" + str(pre_run_epochs))
load_model(inf_model, main_path + "inf_model_ep" + str(pre_run_epochs))

e_lr = 0.0001
g_lr = 0.0001
i_lr = 0.0001
e_decay = 0.00001
g_decay = 0.
i_decay = 0.
e_beta1 = 0.5
e_beta2 = 0.999
g_beta1 = 0.5
g_beta2 = 0.999
i_beta1 = 0.5
i_beta2 = 0.999
#optE = torch.optim.Adam(e_func.parameters(),    lr=e_lr, weight_decay=e_decay, betas=(e_beta1, e_beta2))
#optG = torch.optim.Adam(gen_model.parameters(), lr=g_lr, weight_decay=g_decay, betas=(g_beta1, g_beta2))
#optI = torch.optim.Adam(inf_model.parameters(), lr=i_lr, weight_decay=i_decay, betas=(i_beta1, i_beta2))

print("model E located on: ", next(e_func.parameters()).device)
print("model G located on: ", next(gen_model.parameters()).device)
print("model I located on: ", next(inf_model.parameters()).device)

### Training the model
e_func.eval()
gen_model.eval()
inf_model.eval()


## 3.1 Validation (scores)
    
x_real  = []
x_gen   = []
x_gen_tensor   = []
reconl  = 0
reconl1 = 0
reconl2 = 0
reconl3 = 0
reconl4 = 0
reconl5 = 0
reconl6 = 0
en_pos  = 0
en_neg  = 0
en_noi  = 0
num     = 0

for v_idx, (val_x_real, labels) in enumerate(test_loader): # iterate batches
    #print("## Val. batch ",  v_idx*m , "/", len(test_loader.dataset))
    n = val_x_real.shape[0]
    val_z_real, _, _ = inf_model.forward(val_x_real.to(device))
    val_z_noise      = torch.randn(val_z_real.size()).to(device)
    val_z_fake       = p_sample_progressive(val_z_noise, e_func)

    val_z_diff_1     = q_sample(val_z_real, torch.ones(n, dtype=int).to(device) * 2).detach()
    val_z_diff_2     = q_sample(val_z_real, torch.ones(n, dtype=int).to(device) * 4).detach()
    val_z_diff_3     = q_sample(val_z_real, torch.ones(n, dtype=int).to(device) * 6).detach()
    val_z_diff_4     = q_sample(val_z_real, torch.ones(n, dtype=int).to(device) * 8).detach()
    val_z_diff_5     = q_sample(val_z_real, torch.ones(n, dtype=int).to(device) * 10).detach()
    val_z_diff_6     = q_sample(val_z_real, torch.ones(n, dtype=int).to(device) * 12).detach()

    val_z_recon_1    = p_sample_progressive(val_z_diff_1, e_func, 2)[0, :, :]
    val_x_recon_1    = gen_model(val_z_recon_1.view(-1, latent_dim, 1, 1).float()).detach().cpu()
    del val_z_recon_1, val_z_diff_1
    val_z_recon_2    = p_sample_progressive(val_z_diff_2, e_func, 4)[0, :, :]
    val_x_recon_2    = gen_model(val_z_recon_2.view(-1, latent_dim, 1, 1).float()).detach().cpu()
    del val_z_recon_2, val_z_diff_2
    val_z_recon_3    = p_sample_progressive(val_z_diff_3, e_func, 6)[0, :, :]
    val_x_recon_3    = gen_model(val_z_recon_3.view(-1, latent_dim, 1, 1).float()).detach().cpu()
    del val_z_recon_3, val_z_diff_3
    val_z_recon_4    = p_sample_progressive(val_z_diff_4, e_func, 8)[0, :, :]
    val_x_recon_4    = gen_model(val_z_recon_4.view(-1, latent_dim, 1, 1).float()).detach().cpu()
    del val_z_recon_4, val_z_diff_4
    val_z_recon_5    = p_sample_progressive(val_z_diff_5, e_func, 10)[0, :, :]
    val_x_recon_5    = gen_model(val_z_recon_5.view(-1, latent_dim, 1, 1).float()).detach().cpu()
    del val_z_recon_5, val_z_diff_5
    val_z_recon_6    = p_sample_progressive(val_z_diff_6, e_func, 12)[0, :, :]
    val_x_recon_6    = gen_model(val_z_recon_6.view(-1, latent_dim, 1, 1).float()).detach().cpu()
    del val_z_recon_6, val_z_diff_6
    
    val_x_hat_real   = gen_model(val_z_real.view(-1, latent_dim, 1, 1)).detach().cpu()
    val_x_hat_fake   = gen_model(val_z_fake[0, :, :].view(-1, latent_dim, 1, 1)).detach().cpu()
    # print("val_z_real ", val_z_real.size())
    # print("val_z_real ", val_z_fake.size())
    # print("val_x_hat_fake ", val_x_hat_fake.size())

    recon_loss       = mseloss(val_x_hat_real, val_x_real) / (img_shape[-2]*img_shape[-1])
    recon_loss_1     = mseloss(val_x_recon_1, val_x_real) / (img_shape[-2]*img_shape[-1])
    recon_loss_2     = mseloss(val_x_recon_2, val_x_real) / (img_shape[-2]*img_shape[-1])
    recon_loss_3     = mseloss(val_x_recon_3, val_x_real) / (img_shape[-2]*img_shape[-1])
    recon_loss_4     = mseloss(val_x_recon_4, val_x_real) / (img_shape[-2]*img_shape[-1])
    recon_loss_5     = mseloss(val_x_recon_5, val_x_real) / (img_shape[-2]*img_shape[-1])
    recon_loss_6     = mseloss(val_x_recon_6, val_x_real) / (img_shape[-2]*img_shape[-1])
    
    x_real  += [val_x_real.detach()]
    x_gen   += [val_x_hat_fake.detach()]
    x_gen_tensor   += [val_x_hat_fake]
    reconl  += recon_loss.detach()
    reconl1 += recon_loss_1.detach()
    reconl2 += recon_loss_2.detach()
    reconl3 += recon_loss_3.detach()
    reconl4 += recon_loss_4.detach()
    reconl5 += recon_loss_5.detach()
    reconl6 += recon_loss_5.detach()
    num     += n

    if num > 100:
        break

x_gen  = np.moveaxis(np.concatenate(x_gen), 1, -1)
x_real = np.moveaxis(np.concatenate(x_real), 1, -1)


x_gen_tensor = torch.cat(x_gen_tensor)
chosen_gen = x_gen_tensor[np.random.choice(x_gen.shape[0], 64), :, :, :]
print(chosen_gen.shape)
#print("haha")
"""
s = imgs_per_step.shape[1][-1]

#step_size = callback.num_steps // callback.vis_steps
imgs_to_plot = imgs_per_step[step_size-1::step_size,s]
imgs_to_plot = torch.cat([imgs_per_step[0:1,s],imgs_to_plot], dim=0)
grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1), pad_value=0.5, padding=2)
grid = grid.permute(1, 2, 0)
"""
def plot(p, x):
    return torchvision.utils.save_image(torch.clamp(x, -1., 1.), p, normalize=True, nrow=8)

plot(main_path+"/samples.png", chosen_gen)
