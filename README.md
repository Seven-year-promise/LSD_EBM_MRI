## <div align="center">Energy-Based Prior Latent Space Diffusion model for Reconstruction of Lumbar Vertebrae from Thick Slice MRI [DGM4MICCAI 2024]</div>

[**Yanke Wang**](https://scholar.google.com/citations?user=BdZ513MAAAAJ&hl=en) · [**Yolanne Y. R. Lee**]() · [**Aurelio Dolfini**]() · [**Markus Reischl**](https://www.iai.kit.edu/english/921_1187.php) · [**Ender Konukoglu**]()· [**Kyriakos Flouris**]()

<a href="[https://arxiv.org/abs/2404.00815](https://arxiv.org/abs/2412.00511)"><img src='https://img.shields.io/badge/arXiv-2404.00815-red?logo=arXiv' alt='arXiv'></a>


![lsd_ebm](https://github.com/user-attachments/assets/c206dead-e210-431e-ade0-628fc2204ce3)

The overall architecture of the \ac{lsdebm} is visualized in the figure. Given an input 3D image $`\mathbf{x}`$, the inference network generates the latent variable $`\mathbf{z}_0 \sim q_\varphi(\mathbf{z}_0|\mathbf{x})= \mathcal{N}(\mathbf{z};\mu_0(x),\sigma_0(x))`$ with learnable mean $`\mu_0`$ and variance $`\sigma_0`$. A latent diffusion and denoising processes are constructed with the energy-based prior to optimize $`\mathbf{z}_0`$. The diffusion in latent space acts as checkpoints guiding the learning while also reducing its computational overhead which would be prohibitive in full image space, therefore resulting in more stable and accurate generation. The optimized $`\mathbf{z}_0`$ is then used by the generation network to reconstruct the 3D image $`\mathbf{x}^\prime \sim p_\beta(\mathbf{x}|\mathbf{z}_0)`$.

## <div align="center">LSD-EBM 🚀 NEW</div>

### 

Our work is accepted by 4th MICCAI Workshop, DGM4MICCAI 2024, which will be Held in Conjunction with MICCAI 2024, Marrakesh, Morocco, October 10, 2024. Feel free to contact and have discussions during the event.


## <div align="center">Documentation</div>

LSD-EBM 🚀 is based on a latent space diffusion energy-based prior to leverage diffusion models, which exhibit high-quality image generation for both in 2D and 3D formats. The performance of the those models are tested on the 2D public datasets (MNIST, CIFAR10, and CelebA), and our domestic vertebrae datasets (3D). 

We hope that this work will help the researchers and engineers in image generation or out-painting for MRI imaging. 

To request an access to the mentioned MRI dataset in the article, please free to contact us.

<details open>
<summary>Installation</summary>
 
Create the environment with [**Python>=3.7**](https://www.python.org/) using `ve_py37_ebm.yml` for EBM models and `ve_py37_lsd_ebm.yml` for LEBM and LSD-EBM models


Install the packages required in `requirements.txt`.


```bash
pip install -r requirements.txt
```


</details>

<details open>

<summary>Usage</summary>

#### 1. Run the codes on 2D datasets

```
cd 2D
```

Run the EBM on 2D datasets and the FID scores as well as geberated samples are generated automatically.

```
cd EBM
python ebm_mnist.py
python ebm_cifar10.py
python ebm_celeba64.py
python ebm_fashion_mnist.py
```

Run the LEBM on 2D datasets and the FID scores as well as geberated samples are generated automatically.

```
cd LEBM
python train_mnist.py
python train_cifar10.py
python train_celeba64.py
python train_fashion_mnist.py
```

Run the LSD-EBM on 2D datasets

```
cd LSD-EBM
python LDEBM_Datasets_NEW_CelebA.py
python LDEBM_Datasets_NEW_CIFAR10.py
python LDEBM_Datasets_NEW_Fashion_MNIST.py
python LDEBM_Datasets_NEW_MNIST.py
```

To generate the FID scores and geberated samples for LSD-EBM, run

```
cd LSD-EBM
python LDEBM_Datasets_NEW_CelebA_Inference.py
python LDEBM_Datasets_NEW_CIFAR10_Inference.py
python LDEBM_Datasets_NEW_FashionMNIST_Inference.py
python LDEBM_Datasets_NEW_MNIST_Inference.py
```

#### 2. Run the codes on 3D datasets

```
cd 3D/LSD-EBM/LSD-EBM_code
```

Run the VAE on 3D datasets

```
python train_vae_vert.py
```

Run the LEBM on 3D datasets

```
python train_lebm_vert.py
```

Run the LSD-EBM on 3D datasets

```
python train_lsd_ebm_vert.py
```
</details>

## <div align="center">Evaluation and Analysis</div>


#### 1. Compare the results of different steps of LSD-EBM as the figure shows

```
python compare_steps_lsd_ebm.py
```

![lq2hq](https://github.com/user-attachments/assets/b14585fb-4575-4590-b8c0-1fe2bb5bbcbc)

#### 2. Compare the latents of LSD-EBM and LEBM 

```
python compare_latent_lsdebm_ddpm.py
```

![latentanalysis](https://github.com/user-attachments/assets/a29c1daf-6ab3-4890-afa1-96a316dfb52d)

#### 3. Generate the reconstruction metrics used in the paper

```
python reconstruction_metrics.py
```


|Method | DICE | VS | SEN | SPEC | NMI | CK |
| :---: | :---:|:---: |:---: |:---: |:---: |:---: |
VAE  |  0.7626   ($\pm$ 0.0457) | 0.7887 ($\pm$ 0.0448) | 0.9667 ($\pm$ 0.0138) | 0.9882 ($\pm$ 0.0026) | 0.6252 ($\pm$ 0.0451) | 0.7566 ($\pm$ 0.0461) |
LEBM  | 0.7619 ($\pm$ 0.0576) | 0.7866 ($\pm$ 0.0539)  |  0.9692 ($\pm$ 0.0610) |  0.9883 ($\pm$ 0.0026) | 0.6304 ($\pm$ 0.0663) |  0.7560 ($\pm$ 0.0583) |     
LSD-EBM  |  0.8304 ($\pm$ 0.0317) |  0.8627 ($\pm$ 0.0313)  | 0.9625 ($\pm$ 0.0135) |  0.9914 ($\pm$ 0.0020) |  0.6973 ($\pm$ 0.0367) |  0.8258 ($\pm$ 0.0321) |  


## <div align="center">Contribution and Acknowledgement</div>

We would like to give truly thanks to our co-author, Aurelio dolfinis, who origincally created the code containing the implementation of LEBM, and our LSD-EBM for MRI imaging. 

## <div align="center">Citation</div>

```
@InProceedings{10.1007/978-3-031-72744-3_3,
author="Wang, Yanke
and Lee, Yolanne Y. R.
and Dolfini, Aurelio
and Reischl, Markus
and Konukoglu, Ender
and Flouris, Kyriakos",
editor="Mukhopadhyay, Anirban
and Oksuz, Ilkay
and Engelhardt, Sandy
and Mehrof, Dorit
and Yuan, Yixuan",
title="Energy-Based Prior Latent Space Diffusion Model for Reconstruction of Lumbar Vertebrae from Thick Slice MRI",
booktitle="Deep Generative Models",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="22--32",
isbn="978-3-031-72744-3"
}
```
