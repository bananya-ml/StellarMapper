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

Navigate to the directory with the training examples:

```
$ cd ./src/training
```

and following the notebook on how to set-up the model according to your data and needs.

## Data

The `datasets` directory contains organized subdirectories named after the origin of the data. Each subdirectory contains a separate `README` detailing how the data was obtained and its usage.

## Implemented Models

|Model   |Title                                                                            |
|--------|---------------------------------------------------------------------------------|
|StarNet |An application of deep learning in the analysis of stellar spectra<sup>[1]</sup>(#sfabbro)|
|STARCNET|StarcNet: Machine Learning for Star Cluster Identification<sup>[2]</sup>(#gperez)         |

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## References

- <a name='sfabbro'></a>[S Fabbro, K A Venn, T O'Briain, S Bialek, C L Kielty, F Jahandar, S Monty, An application of deep learning in the analysis of stellar spectra, Monthly Notices of the Royal 
  Astronomical Society, Volume 475, Issue 3, April 2018, Pages 2978–2993](https://doi.org/10.1093/mnras/stx3298)
- <a name="gperez"></a>[Pérez, Gustavo and Messa, Matteo and Calzetti, Daniela and Maji, Subhransu and Jung, Dooseok E. and Adamo, Angela and Sirressi, Mattia, StarcNet: Machine Learning for Star Cluster 
  Identification, The Astrophysical Journal](https://iopscience.iop.org/article/10.3847/1538-4357/abceba)
- <a name="mpant"></a>[M Pantaleoni González, J Maíz Apellániz, R H Barbá, B Cameron Reed, The Alma catalogue of OB stars – II. A cross-match with Gaia DR2 and an updated map of the solar neighbourhood, Monthly 
  Notices of the Royal Astronomical Society, Volume 504, Issue 2, June 2021, Pages 2968–2982](https://doi.org/10.1093/mnras/stab688)
