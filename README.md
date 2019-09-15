# ERPENet (Multi-task Autoencoder) for P300 EEG-Based BCI
model.py -- contains all model builders in Keras.  
train.py -- used to train the models. log file, tensorboard file, and best weights are kept.  
benchmark.py -- used to evaluate the trained model; need .hdf5(weight) from the train.py file as one of the input.  
X_dawn -- xDawn algorithm as one of the baseline.  
