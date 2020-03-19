# Urban Sound Classification


## Introduction

Sonic event classification is a field of growing research. Most of these researches focuses on music or speech recognition. There are scarce works on environment sounds with very few databases for labeled environment audio data.

The objectives of this project is to evaluate and train various machine learning models to classify the urban sounds into categories correctly. 

## Data Set

This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The classes are drawn from the urban sound taxonomy. For a detailed description of the dataset and how it was compiled please refer to our paper.

All excerpts are taken from field recordings uploaded to www.freesound.org. The files are pre-sorted into ten folds (folders named fold1-fold10) to help in the reproduction of and comparison with the automatic classification results reported in the article above.

In addition to the sound excerpts, a CSV file containing metadata about each excerpt is also provided.

Download Link: [Urban Sound 8K Audio Dataset](https://urbansounddataset.weebly.com/)

## Feature Extraction

We used the VGG-like model to generate the 128-dimensional features using the [audioset](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) library available in the TensorFlow models Github repository, along with supporting code for audio feature generation, embedding postprocessing, and demonstrations of the model in inference and training modes.

[View Feature Extraction Here](https://github.com/rajinigurijala/Capstone4/blob/master/ExtractFeatures.ipynb)


## Model Evaluation

We will evaluate 4 different models in this project – Support Vector Machine, Random Forest, Deep Neural Networks and Convolutional Neural Networks. 

The preprocessed audio data with 896 dimensions and 8275 observations is split into 80% training data set and 20% test data set. The data has been scaled using the Standard Scaler prior to modeling.

- Both Support Vector Machine and Random Forest models hyperparameters were tuned using GridSearchCV. And then these 2 models were trained on training dataset using the best parameters.
[View Final Capstone Jupyter Notebook Here](https://github.com/rajinigurijala/FinalCapstone/blob/master/UrbanSound_Final_Capstone.ipynb)

- We used the Artificial Neural Networks to train the DeepNN model and evaluated. In this model GridSearchCV has been used for Hyperparamter optimization.
[View DeepNN using GridSearch](https://github.com/rajinigurijala/FinalCapstone/blob/master/UrbanSoundKeras_GridSearch.ipynb) 

- Alternatively we created additional DeepNN model using Keras Tuner for optimizing and tuning the hyperparameters.
[View DeepNN using Keras Tuner](https://github.com/rajinigurijala/FinalCapstone/blob/master/UrbanSoundKeras_Tuner.ipynb)

- Finally we trained CNN model using SciKit Optimize. We reshaped the 1-dimenstional (896 attributes) dataset to 2-Dimensional (64x14) so that the CNN model can be trained and tested.
[View CNN with using Kit Optimize](https://github.com/rajinigurijala/FinalCapstone/blob/master/UrbanSoundKeras_CNN.ipynb)

The supervised machine learning and newural network models were evaluated using the accuracy scores and confusion matrix to select best performing model.

## Project Report

View: [Urban Sound Classification Project Report](https://github.com/rajinigurijala/FinalCapstone/blob/master/Audio%20Classification%20Project%20Report.pdf)

## References

1. [VGGish Audio Embedding Colab](https://colab.research.google.com/drive/1TbX92UL9sYWbdwdGE0rJ9owmezB-Rl1C#scrollTo=DaMrmOEvC7L4)
2. [Hyperparameter Optimization in TensorFlow](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb)
3. [Hyperparameter Tuning using Keras Tuner](https://www.sicara.ai/blog/hyperparameter-tuning-keras-tuner)


## Technologies Used
- Python, Numpy, Pandas, Matplotlib, Seaborn, SKLearn, Librosa, Audioset
- Jupyter Notebook
