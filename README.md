# Handwritten Digit Generation

## 1. Handwritten Digit Generation with VAE

### Description: 
In this project, we will implement a VAE, where the encoder is an LSTM network and the decoder is a convolutional network.
We will use the MNIST dataset as our train data. 

**Encoder:**

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.53.57.png)

This figure shows the encoder part of the VAE model. It is a single layer LSTM (with default
configurations e.g. activation is tanh) whose output connected to two dense layers for mean and log
variance of latent space distribution. Images are treated as 28 dimensional 28 timestamp inputs.The
dimensionality of the hidden states is 128. Latent space dimensionality is 50. To generate the input image, we define a custom resampling layer where a random vector is generated from the latent space  based on the mean and variance layers and Gaussian distribution, and then it is feeded into decoder.  

**Decoder** :

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.54.51.png)

This figure shows the decoder architecture.It takes a vector of size 50 (latent space vector) and
outputs 28x28 images. To create images from vectors, first we reshape dense layer output then apply
transposed convolutions with 2 strides which doubles the image size and relu activation function.
The last transposed convolution layer has sigmoid activation to create pixel values.

**Loss function:**

Loss function consists of two parts: reconstruction loss with binary cross-entropy and the
regularization term with KL-divergence. The closed-form KL divergence formula for normal
distribution is used.

**Training:**

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.55.12.png)

The model is trained with 30 epochs. The first figure shows the total loss function. We can see it
decreases fast in the first few epochs and then decreases slowly towards epoch 30. The second figure
displays binary cross-entropy loss function to reconstruct the image. Its behaviour is pretty similar to
total loss function with fast decrease followed by small changes. The final plot shows the KL loss and
it is interesting that it increase throughout the training although the overall loss decreases with the
domination of reconstruction loss. Although we desire the KL loss to decrease as well, we saw this
kind of behaviour also in the examples on internet. KL loss has regularization effect by bringing latent
space distribution close to Gaussian distribution and prevents overfitting like auto encoders. So, this
behaviour may not be a huge problem for VAEs, at least we should comment on this after looking at
the generated images.

**Note** : The loss function plots are obtained from tensorboard and they are applied exponential
smoothing by default. This is useful to see the trend in a fluctuated loss graphs (we will especially see
this in GAN models below)

**Generation:**

We generated 100 images from 100 random vectors of size 50. The same random vectors are used
throughout the images shown in this report. Results can be seen below. Most of the generated
images looks similar to MNIST digists with some of them are like combination of two digits. It is
possible if the random point is somewhere between mean vectors of two digits in latent space. It
seems all digits are available in generated images but some of are more frequent than others like
digit 9.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.55.35.png)

## 2. Handwritten Digit Generation with GAN

**Generator:**

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.56.12.png)

The figure above shows the architecture of the default generator network. It takes a vector of size
50 and output 28x28 images. We intentionaly make this network very similar to decoder of the VAE
to compare them fairly. We again have transposed convolution layers with stride 2. In this case, we
have batch normalization after linear output and leaky relu activation function, as recommended in
the GAN literature.

The figure below displays the architecture of the more complex generator network. The only two
difference are the dense layer dimension and the number of feature maps in transposed convolution
layers. Feature maps doubles in the below network compared to above one. The number of
parameters increase from 98832 to 225312.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.56.34.png)

**Discriminator:**

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.56.57.png)

The discriminator goes in the opposite direction of the generator. It takes an 28x28 image and apply
convolution layers with stride 2. In this case, we have dropout layers after applying activations. We
trained our models with cross-entropy loss and Wasserstein loss.


**Result 1: GAN with default generator and cross entropy loss**

The figures below shows the cross-entropy loss function of GAN with default generator. The first one
belogns to discrminator and the other is generator’s. We trained 50 epochs We clearly see the
adverserial behaviour in the loss functions. At the beginning, generator learns faster and
discriminator loss increases dramatically. After first few epochs, discriminator starts improving in a
fast manner and makes generator loss increase until around epoch 10. Then generator again makes
an attack and its loss decrease over a long time with some up and downs. At the end, actually we
dont get a converged point but it takes quite long time to train this models with small GPUs (around
2 - 2.5 hours in this configuration) so we stop the training.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.57.14.png)

We stored some checkpoints during training and obtained following images from same random
vectors.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.57.33.png)

The 100 images generated from this GAN model are as follows.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.57.51.png)

**Result 2: GAN with more complex generator and cross entropy loss**

The figures below shows the cross-entropy loss function of GAN with more complex generator. The
first one belogns to discriminator and the other is generator’s. We trained 50 epochs We clearly see
the adverserial behaviour in this loss functions as well. After some epochs, discriminator learns faster
and generator loss increases dramatically. After 20 - 25 epochs, generator starts improving in a fast
manner and makes discriminator loss increase. At the end, generator is a bit converged but
discriminator has larger fluctiations.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.58.14.png)

We stored some checkpoints during training (10,30,50 epochs) and obtained following images from
same random vectors.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.58.25.png)

The 100 images generated from this GAN model are as follows.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.58.52.png)

**Result 3: WGAN with default generator**

The figures below shows the Wassersteinloss function of GAN with default generator. The first one
belongs to discriminator and the other is generator’s. We trained 30 epochs We clearly see the
adverserial behaviour in this loss functions as well. At the beginning, discriminator learns faster and
generator loss increases. After that, generator starts improving in a small period then degrades until
20 epochs. At the end, generator is improving and dicriminator degrades, not converged due to
limited epochs.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.59.11.png)

We stored some checkpoints during training (10,20,30 epochs) and obtained following images from
same random vectors.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2022.59.24.png)

The 100 images generated from this GAN model are as follows.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2023.00.27.png)

**Result 4: WGAN with more complex generator**

The figures below shows the Wasserstein loss function of GAN with more complex generator. The
first one belongs to discriminator and the other is generator’s. We trained 30 epochs We clearly see
the adverserial behaviour in this loss functions as well. At the beginning, generator learns faster and
discriminator loss increases over a long period untill around 20 epochs. After that, generator starts
degrading and discriminator fluactuates. At the end, they are not converged yet due to limited
epochs

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2023.06.20.png)

We stored some checkpoints during training (10,20,30 epochs) and obtained following images from
same random vectors.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2023.06.39.png)

The 100 images generated from this GAN model are as follows.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2023.07.00.png)

For quantitative evaluation, we used Inception Scrore as it is well adopted in the literature. The table
below shows the sample mean and standard deviaton of inceptions scores take from 10 sample
groups as a recommended practice. We used a CNN based classifier rather than inception network
since we have one channel image and different classes.

![](https://github.com/omerfarukcavass/Handwritten-Digit-Generation/blob/main/images/Ekran%20Resmi%202022-09-18%2023.07.14.png)

The bigger inception score means better score. Although VAE images look closer to MNIST digits
when evaluated manually, GAN images have better inception scores. The VAE images have some blur
effects, which may be the reason for worse performance.

**References:**

https://keras.io/examples/vision/mnist_convnet/

https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-
evaluating-generated-images/

https://www.geeksforgeeks.org/role-of-kl-divergence-in-variational-autoencoders/

https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/

https://www.tensorflow.org/tutorials/generative/dcgan

https://www.tensorflow.org/


