# Image2StyleGAN

This is an unofficial reimplementation. 

Dirfferences from the official Image2StyleGAN
1. Reframe it to optimizing in StyleGAN v2.
2. optimize the shift w+ code from the mean w+. 

## Setup
- **Requirements**
```
Pytorch 1.4.0
Tensorflow 1.14
gcc>5
requests
tqdm
```

- **StyleGAN2-Pytorch**
Please First replace `dnnlib` of the official stylegan2 code with this in my repo. And then convert model as the guidence of [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).

## Get Start.
1. Align images. 
    * Download dlib: 
        ```
        wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
        ```
    * run `crop.sh`

2. Inversion
    run `invert.sh`
    

### Refernces
We benifit from some great works.
1. [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
2. [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
