# TopoRET
Code for the paper "Topology-Enhanced Transformers for Retinal Disease Diagnosis" <br />

Our project explores the application of topological methods with deep learning to improve retinal disease diagnosis. We utilize topological data analysis in the form of Betti vectors to create more robust and accurate models to classify these diseases. <br /> 

## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Data](#data)
* [Results](#results)
* [Model](#model) 
* [Acknowledgements](#acknowledgements)





## Installation
Please refer to the requirements.txt file to install the dependencies

## Usage
To run each python file, please replace each path with your own. <br />

Betti vectors should be provided in csv format and 400 dimensional in length with an additional 'labels' column <br />
Images should be 128x128 and stored in an npz with an 'images' array 



## Data
Due to file size limitations, we cannot provide the dataset or betti vectors in this repository. However, the link to the original dataset can be found below: <br />
* [Eye Disease Dataset](https://data.mendeley.com/datasets/s9bfhswzjb/1)

## Results

The more detailed results of our experiments can be found in our paper. To summarize, topological features often enhance the performance of baseline models accross all classification tasks.

![performance_plot1 (updated) 1](https://github.com/user-attachments/assets/9e977c8f-b695-48b9-a241-ddc298371db9)

## Model 

The high level architecture of our model can be found below

![Topovision](https://github.com/user-attachments/assets/05abfc27-833a-4afd-aa2e-cc550c204f1d)


## Acknowledgements 
* We would like to thank the authors of the [data](#data) for their time and effort in developing this valuable resource.
