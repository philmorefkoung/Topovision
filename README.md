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
To run each python file in the models folder, please download the environment.yml file and in the python file replace the paths to the data with your own. <br />
Using conda (tested on version 24.1.2) run the following commands:
```
cd path/to/files
conda env create -f environment.yml
conda activate toporet
python PH+Swin.py
```

## Usage
Betti vectors should be provided in csv format and 400 dimensional in length with an additional 'labels' column <br />
Images should be 128x128 and stored in an npz with an 'images' array 



## Data
Due to file size limitations, we cannot provide the data in this repository. However, the link to the original dataset can be found below: <br />
* [Eye Disease Dataset](https://data.mendeley.com/datasets/s9bfhswzjb/1)
* [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
* [ORIGA](https://pubmed.ncbi.nlm.nih.gov/21095735/)
## Results

The more detailed results of our experiments can be found in our paper. To summarize, topological features often enhance the performance of baseline models accross all classification tasks.

![performance_plot1 (updated) 1](https://github.com/user-attachments/assets/9e977c8f-b695-48b9-a241-ddc298371db9)

## Model 

The high level architecture of our model can be found below

![Topovision](https://github.com/user-attachments/assets/05abfc27-833a-4afd-aa2e-cc550c204f1d)


## Acknowledgements 
* We would like to thank the authors of the [data](#data) for their time and effort in developing this valuable resource.
