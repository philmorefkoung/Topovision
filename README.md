# TopoRET
Code for the paper "Topology-Enhanced Transformers for Retinal Disease Diagnosis" <br />

Our project explores the application of topological methods with deep learning to improve retinal disease diagnosis. We utilize topological data analysis in the form of Betti vectors to create more robust and accurate models to classify these diseases. <br /> 

## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Data](#data)
* [Results](#results)
* [Acknowledgements](#acknowledgements)





## Installation
Please refer to the requirements.txt file to install the dependencies

## Usage
Provided below is an example to load the data: <br />
```
import numpy as np
import pandas as pd

image_train = np.load('babesia-train.npz')['images'] 
image_val = np.load('babesia-val.npz')['images'] 
image_test = np.load('babesia-test.npz')['images'] 

betti_train = pd.read_csv('babesia-betti-train.csv')
betti_val = pd.read_csv('babesia-betti-val.csv')
betti_test = pd.read_csv('babesia-betti-test.csv')

labels_train = np.load('babesia-train.npz')['labels']
labels_val = np.load('babesia-val.npz')['labels']
labels_test = np.load('babesia-test.npz')['labels']
```
Betti vectors should be provided in csv format and 400 dimensional in length with a 'labels' column <br />
Images should be stored in an npz with an 'images' array 



## Data
Due to file size limitations, we cannot provide the dataset or betti vectors in this repository. However, the link to the original dataset can be found below: <br />
* [Eye Disease Dataset](https://data.mendeley.com/datasets/s9bfhswzjb/1)

## Results

The specific results of our experiments can be found in our paper. To summarize, topological features enhanced the performance of baseline models accross all classification tasks.



## Acknowledgements 
* We would like to thank the authors of the [data](#data) for their time and effort in developing this valuable resource.
