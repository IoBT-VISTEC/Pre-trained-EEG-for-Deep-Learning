# ERPENet (Multi-task Autoencoder) for P300 EEG-Based BCI
The event-related potential encoder network (ERPENet) is a multi-task autoencoder-based model, that can be applied to any ERP-related tasks. 

For more details, please refer to: https://ieeexplore.ieee.org/abstract/document/8723080


## Code Description (To be updated) ##
model.py -- contains all model builders in Keras.  
train.py -- used to train the models. log file, tensorboard file, and best weights are kept.  
benchmark.py -- used to evaluate the trained model; need .hdf5(weight) from the train.py file as one of the input.  
X_dawn -- xDawn algorithm as one of the baseline.  


## Citation ##
Following citation format can be used for BibTex:

    @ARTICLE{8723080,
    author={A. {Ditthapron} and N. {Banluesombatkul} and S. {Ketrat} and E. {Chuangsuwanich} and T. {Wilaiprasitporn}},
    journal={IEEE Access},
    title={Universal Joint Feature Extraction for P300 EEG Classification Using Multi-Task Autoencoder},
    year={2019},
    volume={7},
    pages={68415-68428},
    doi={10.1109/ACCESS.2019.2919143},
    }
