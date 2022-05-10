# cpsc452-project

This is our final project code repository for Yale's Deep Learning (CPSC 452) class. We train a convolutional variational autoencoder to learn a latent representation of face images and use it to alleviate bias in face recognition. The model architecture is detailed below: 

<p align="center">
<img src="./Model.png" width="480">
</p>

The main model training code is located in `Deep_Learning_Training.ipynb`. We use data from [UTKFace](https://susanqq.github.io/UTKFace/). All of our code is written in PyTorch. For more information please consult our paper, [**Debiasing Face Detection on Convolutional Variational Autoencoders**].
