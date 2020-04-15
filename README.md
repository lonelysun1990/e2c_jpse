# e2c_jpse
Code example for "deep-learning-based surrogate model for reservoir simulation with time-varying well controls" on JPSE.

Zhaoyang Larry Jin, Yimin Liu, Louis J. Durlofsky

Journal of Petroleum Science and Engineering
https://doi.org/10.1016/j.petrol.2020.107273

## How to cite:  
@article{jin2020deep,
  title={Deep-learning-based surrogate model for reservoir simulation with time-varying well controls},
  author={Jin, Zhaoyang Larry and Liu, Yimin and Durlofsky, Louis J},
  journal={Journal of Petroleum Science and Engineering},
  pages={107273},
  year={2020},
  publisher={Elsevier}
}



This workflow is tested with Tensorflow 1.10.0 (cpu/gpu).


## 1. prepare_e2c_training_data.ipynb  
Prepare the data for e2c training process. Here we assume that the simulation data (output of commercial simulator) is ready. The purpose of this step is re-orgainze the data so that it can easily consumed by the E2C model in the following step, which includes spliting the data into training set and test set.

## 2. e2c_train.ipynb  
Construct the E2C model with the training dataset.

## 3. e2c_eval.ipynb  
Run the E2C model on the test dataset.

