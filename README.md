# e2c_jpse
Code example for "deep-learning-based surrogate model for reservoir simulation with time-varying well controls" on JPSE.

Zhaoyang Larry Jin, Yimin Liu, Louis J. Durlofsky

Journal of Petroleum Science and Engineering  
https://doi.org/10.1016/j.petrol.2020.107273

Data is available at:  
https://drive.google.com/drive/folders/1P-R6uNkzw4lbVjgOIoe42okom08MtAN7?usp=sharing

## How to cite:  
```
@article{jin2020deep,  
  title={Deep-learning-based surrogate model for reservoir simulation with time-varying well controls},  
  author={Jin, Zhaoyang Larry and Liu, Yimin and Durlofsky, Louis J},  
  journal={Journal of Petroleum Science and Engineering},  
  pages={107273},  
  year={2020},  
  publisher={Elsevier}  
}
```


This workflow is tested with Tensorflow 2.5.0 (cpu/gpu).


## 1. prepare_e2c_data.ipynb  
Prepare the data for e2c training process. Here we assume that the simulation data (output of commercial simulator) is ready. The purpose of this step is re-orgainze the data so that it can easily consumed by the E2C model in the following step, which includes spliting the data into training set and test set.

## 2. e2c_train_and_test.ipynb  
Construct the E2C model.  
Train E2C with the training dataset.  
Evaluate E2C and provide basic plots.  


