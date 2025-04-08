1.Dataset preparation
For SUPPORT and SYNTHETIC dataset, please refer to auton survival's setting
The links are provided: https://github.com/autonlab/auton-survival
For MIMIC-III and MIMIC-IV, please refer to the github repo for preprocessing steps:
https://github.com/YerevaNN/mimic3-benchmarks
we extract time-series data for 17 clinical variables from the first 48 hours of each ICU admission


2.Environment setting

conda env create -f environment.yml

3.Run model
use the DDPSurv Branch 

python baseline.py 

add arguments by your own requirements
