---
title:  Recognizing Korean Food in Photos through Transfer Learning of Deep Features

date: March 7, 2016

author: Joseph Kim

bibliography: references.bib

geometry: left=1in,right=1in,top=1in,bottom=1in

csl: ieee-with-url.csl
...


# Project Overview

Transfer learning in machine learning is the application of knowledge learned in one problem to another similar or related problem, and has been around for at least a couple of decades [@wiki:transfer]. A relatively recent discovery of transfer learning through trained deep neural networks has raised a lot of excitement and opened doors to a lot of applications. Researchers found that the trained weights of nodes in a trained network, particularly the lower and mid-level nodes, can be copied readily to other deep networks, along with the structure of the deep network, and just retraining the high-level classification layer can result in a network that performs very well for a new problem. For example, for image recognition, taking a convolutional neural network (CNN) that performs very well on recognizing various types of images and retraining the classification layer to identify a specific set of labels such as dog breeds or food types can work well. That's because the lower nodes and layers of the CNN learn low-level features of the images, which are necessary for many image recognition tasks, but the higher nodes and layers learn high-level features, which tend to be more specific to the dataset [@YosinskiCBL14].

A big advantage of transfer learning is that we can drastically reduce the total amount of computations needed to train deep networks. Successful deep learning projects are often trained on high performance clusters and servers with GPUS to reduce the time to compute all the vectorized matrix operations, and runtimes can be hours, days and even weeks. With only a personal laptop, an engineer can take one of these trained networks, and just retrain the last layer of the network on a smaller set of data for a specific classification task without performance or memory issues.

In this project, I retrained the top layer of Google's Inception-v3 CNN to recognize 20 kinds of Korean food in photos. Inception-v3 was a very easy choice for a pre-trained network, which had been trained on ILSVRC2014 dataset to perform very well on the ImageNet Large Visual Recognition Challenge [@SzegedyVISW15]. The ILSVRC dataset contains 1.2 million images from 1000 categories [@ILSVRC15], which means that Inception v3 was a good choice for a generic image recognition tool that I can fine-tune for my problem. Also at the time of this writing, it outperforms all the other models for the ImageNet dataset. Therefore this network was used to generate 2048 deep features, and the top fully-connected layer of the network was retrained, which applies the Softmax function to output probability values for each of the Korean food labels.


# References
