# ALS II cross-matched with Gaia DR3

This data was obtained from [Gaia Data Release 3](https://www.cosmos.esa.int/web/gaia/data-release-3) to study if spectra from massive stars can be used to differentiate them from their low-mass counterparts.

## Data Description

There are 3 different subdirectories: `queries`, `RVS` and `XP`. `queries` contains the raw outputs obtained from the Gaia Archive after 

- cross-matching the objects in ALS II
- low mass stars for balancing the dataset
- high-mass stars found by the Gaia survey but not included in the ALS II catalogue. The Gaia survey uses its own Final Luminosity Age Mass Estimator (FLAME) program to determine the object's mass.
- labels as obtained from the ALS II catalogue

`RVS` contains the if-available RVS spectrum of each object, and `XP` contains the BP-RP spectrum of each object, as obtained from the Gaia archive.

## Usage

This data is directly combined and processed for both classification and regression tasks. Stellar parameters T<sub>eff</sub>, log g, and Fe/H can be used for regression tasks, and the 'labels' data can be
used for classification.