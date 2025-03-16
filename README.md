

<a name="readme-top"></a>
<!--





<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://gitlab-student.centralesupelec.fr/alix.chazottes/fmr-2024-segmentation-hierarchique">
    <img src="images/logo_safe.jpg" alt="Logo" width=600>
  </a>

<h3 align="center"> Unofficial version of Fast Learning Signed Distance Functions from Noisy 3D Point Clouds via Noise to Noise Mapping </h3>

  <p align="center">
     Unofficial Pytorch implementation  for Noise 2 Noise in Point cloud  by  Nayoung Kwon and  Pierrick Bournez
    <br />
    <a href="https://gitlab-student.centralesupelec.fr/alix.chazottes/fmr-2024-segmentation-hierarchique"><strong>Explore the docs »</strong></a>

  </p>
</div>

# 📌 Description
This is an repository that try to reproduce the Paper: "Fast Learning Signed Distance Functions from Noisy 3D Point Clouds via Noise to Noise Mapping". 
It is focus on the shape reconstruction part. 
The primary focus was to have preliminary working reproducible results. The implementation  is not intended to be quick but to be able to iterate quickly on it ( and reproducible). 

We used it for a class project, we may add the report here. 

We don't think our results are as good as the paper claim to be, but we hope it will help people to have better result with this method. 

Also, if you ❤️ or simply use this project, don't forget to give the repository a ⭐, it means a lot to us ! 

# 🏗 Getting Started: 
We tested the code with Cuda 12.0 
Download the  requirements : 
```bash
# python3 -m venv venv
# source venv/bin/activate  # for linux
pip install -r requirements.txt
# If you want to use MultiHashResolution uncomment the next line
#pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

```
If the  installation of the fancy InstantNGP models doesn't work, see the [repository](https://github.com/NVlabs/tiny-cuda-nn) for more details. We provided another baseline [SirenNET] to have preliminary results before getting into TinyCUda

# 🚀 Usage
We provide two mains models. One with the Multi Hash Resolution Resolution and [SirenNET] as a comparison. 
We Mostly tested our implementation on 2D Data so we can assess the underlying SDF Data.
See the Notebook ["Main.ipynb"](https://github.com/gardiens/Noise2NoiseMapping_pytorch/blob/main/main.ipynb) for more information.
if you want to scale it to 3D, we provide a proof of concept in the notebook ["3D_version.iypnb"](https://github.com/gardiens/Noise2NoiseMapping_pytorch/blob/main/notebook/3D_version.ipynb) 

We welcome any pull request to make it work on custom ply.


# Repository structure
The repository is structured as follows. Each point is detailed below.
```
├── README.md        <- The top-level README for developers using this project

├── src             <- Source code for use in this project
│   ├── data                      # How we create our synthetic data
│   ├── loss                      # The different losses we implemented.
│   ├── models                    # Model Architecture
│   ├── result                    # Visual Display
│   ├── loss                      # Loss
│   ├── metrics                   # Metrics
│   ├── models                    # Model architecture
│   
│   ├── optimize.py                   # Our training loop
│   
├── requirements.txt  <- The requirements file for reproducing the analysis environment
├── main.ipynb         <- Main script to run the code
└── personal_files <- Personal files, ignored by git (e.g. notes, debugging test scripts, ...)
```


# 💳 Credits
I was inspired by this [documentation](https://github.com/drprojects/superpoint_transformer/tree/master)
The Main structure of the repository come from this  (great) [course](https://jdigne.github.io/mva_geom/)

Official Repository in [Tensorflow](https://github.com/mabaorui/Noise2NoiseMapping). We are not affiliated with this repository and this is our own opinions.

