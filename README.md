# StellarMapper

StellarMapper is a deep learning library created to provide easy and customizable access to SOTA deep learning architectures that might not always 
have provided implementations by the authors. At this moment, StellarMapper is focused on DL models that are introduced in various articles 
in astronomy and astrophysics journals, which often aren't packaged with a detailed explanation of their novel DL methodologies.

The library will also provide 

- utility tools for visualisation and interpretation of the various kinds of data used in astronomy (where solutions are not 
  already available) as well as tools for hyperparameter tuning and model selection.
- in-built model performance monitoring as well as several evaluation metrics.

## Usage

Clone the repository (will add Pypi support when the library becomes extensive enough):

```
$ git clone https://github.com/bananya-ml/StellarMapper
$ cd ./stellarmapper
```

Create a virtual environment:
```
$ python -m venv .venv --upgrade-deps
```

where ".venv" is the name of your virtual environment. Activate the environment and install the required dependencies:

```
$ .\.venv\Scripts\activate
$ pip install -r ./requirements.txt
```

Navigate to the directory with the training examples:

```
$ cd ./src/training
```

and follow the notebook on how to set-up the model according to your data and needs.

+ **NOTE** Currently, I have not tested if the dependencies will hold up on Linux and MacOS machines. Quite likely some changes will be required to install the dependencies on anything 
  other than Windows, so I am also including a rough list of the dependencies:
  - astropy==6.0.0
  - numpy==1.26.2
  - matplotlib==3.8.2
  - pandas==2.1.3
  - torch==2.1.2+cu121
  
## Data

The `datasets` directory contains organized subdirectories named after the origin of the data. Each subdirectory contains a separate `README` detailing how the data was obtained and its usage.

## Implemented Models

|Model   |Title                                                                                                                         |
|--------|------------------------------------------------------------------------------------------------------------------------------|
|StarNet |An application of deep learning in the analysis of stellar spectra<sup>[1](#sfabbro)</sup>                                    |
|STARCNET|StarcNet: Machine Learning for Star Cluster Identification<sup>[2](#gperez)</sup>                                             |
|OTRAIN  |O’TRAIN: A robust and flexible ‘real or bogus’ classifier for the study of the optical transient sky<sup>[3](#kmakhlouf)</sup>|
|AlexNet |One weird trick for parallelizing convolutional neural networks<sup>[4](#kalex)</sup>                                         |
|VGG     |Very Deep Convolutional Networks for Large-Scale Image Recognition<sup>[5](#ksimonyan)</sup>

+ **NOTE** The library contains some redundant architectures like 'AlexNet' and 'VGG' that were added for the sake of having accessibility within the same library. These models are quite popular and might have their official implementations in other libraries (e.g. [torchvision](https://github.com/pytorch/vision/tree/main/torchvision)).



## TODO

❌ Add support for time series data\
❌ Add (extensive) evaluation metrics\
✅ Feature to save and load trained models

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## References

- <a name='sfabbro'></a>[S Fabbro, K A Venn, T O'Briain, S Bialek, C L Kielty, F Jahandar, S Monty, An application of deep learning in the analysis of stellar spectra, Monthly Notices of the Royal 
  Astronomical Society, Volume 475, Issue 3, April 2018, Pages 2978–2993](https://doi.org/10.1093/mnras/stx3298)
- <a name="gperez"></a>[Pérez, Gustavo and Messa, Matteo and Calzetti, Daniela and Maji, Subhransu and Jung, Dooseok E. and Adamo, Angela and Sirressi, Mattia, StarcNet: Machine Learning for Star Cluster 
  Identification, The Astrophysical Journal](https://iopscience.iop.org/article/10.3847/1538-4357/abceba)
- <a name="mpant"></a>[M Pantaleoni González, J Maíz Apellániz, R H Barbá, B Cameron Reed, The Alma catalogue of OB stars – II. A cross-match with Gaia DR2 and an updated map of the solar neighbourhood, Monthly 
  Notices of the Royal Astronomical Society, Volume 504, Issue 2, June 2021, Pages 2968–2982](https://doi.org/10.1093/mnras/stab688)
- <a name="kmakhlouf"></a>[Makhlouf, K. and Turpin, D. and Corre, D. and Karpov, S. and Kann, D. A. and Klotz, A., O’TRAIN: A robust and flexible ‘real or bogus’ classifier for thestudy of the optical transient sky](http://dx.doi.org/10.1051/0004-6361/202142952)
- <a name="kalex"></a>[Alex Krizhevsky, One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)
- <a name="ksimonyan"></a>[Karen Simonyan, Andrew Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)