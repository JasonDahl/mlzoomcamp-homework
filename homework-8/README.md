# Image Classification with Tensorflow and Keras

![Image: A bumblebee](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-8/bumble.png)

## Dataset and Objective
The project aims to build an image classification model to differentiate between bees and wasps. Utilizing the "Bee or Wasp?" dataset from Kaggle, the notebook explores deep learning techniques, specifically Convolutional Neural Networks (CNNs), using TensorFlow and Keras.

## Libraries Utilized
The notebook employs numpy, pandas, matplotlib, seaborn, and TensorFlow with Keras for data processing, visualization, and model development.

## Dataset Overview
The "Bee or Wasp?" dataset, sourced from Kaggle and slightly modified, serves as the foundation for this project's image classification task. This dataset is pivotal in training a model to accurately differentiate between images of bees and wasps. 

### Dataset Structure
- **Total Images:** The dataset consists of approximately 2500 images of bees and around 2100 images of wasps, providing a substantial number of samples for model training and evaluation.
- **Folder Organization:** Images are organized into separate folders, distinguishing between the training and test sets. This organizational structure streamlines the process of data loading and facilitates supervised learning.

### Notable Features
- **Customization:** While the dataset is pre-collected, it was slightly modified, ensuring its suitability for the specific classification task undertaken in this project.
- **GPU Requirement:** Considering the computational complexity involved in training the model from scratch, an environment with GPU capabilities is recommended for optimal performance. However, it's also feasible to execute the code on standard hardware, albeit with slower processing speeds.

## Convolutional Neural Network (CNN) Model
In this project, a Convolutional Neural Network (CNN) architecture is employed to perform image classification tasks. Leveraging Keras, a high-level neural network API, the model is constructed with specific layers and configurations for optimal performance in distinguishing between bee and wasp images.

### Model Structure
- **Input Shape:** The model takes images with a shape of (150, 150, 3) as input, representing the width, height, and color channels.
- **Convolutional Layer:** Utilizing a Conv2D layer with 32 filters, a kernel size of (3, 3), and 'relu' activation, the network performs feature extraction.
- **Max Pooling:** Following the convolutional layer, MaxPooling2D is applied with a pool size of (2, 2) to downsample the feature maps.
- **Flattening:** The results are flattened into vectors for further processing.
- **Dense Layers:** The flattened vectors are passed through a Dense layer with 64 neurons and 'relu' activation, followed by an output Dense layer with 1 neuron and 'sigmoid' activation, suitable for binary classification.
- **Optimizer and Loss Function:** Stochastic Gradient Descent (SGD) with specific parameters (learning rate=0.002, momentum=0.8) is used as the optimizer. Considering the binary classification nature of the task, Binary Crossentropy is chosen as the loss function to optimize model performance.

### Model Evaluation
The compiled model architecture, showcasing various layers, their respective output shapes, and total trainable parameters (11,215,873), is summarized using the `model.summary()` method.

### Notable Features
- **Parameter Count:** The convolutional layer alone comprises 896 parameters, influencing the network's capacity to learn distinct features from the images.
- **Loss Function Selection:** Binary Crossentropy is favored as the most suitable loss function for binary classification tasks due to its effectiveness in optimizing models with sigmoid output activations.

```
Model: "model_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_5 (InputLayer)        [(None, 150, 150, 3)]     0         
                                                                 
 conv2d_4 (Conv2D)           (None, 148, 148, 32)      896       
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 74, 74, 32)       0         
 2D)                                                             
                                                                 
 flatten_4 (Flatten)         (None, 175232)            0         
                                                                 
 dense_8 (Dense)             (None, 64)                11214912  
                                                                 
 dense_9 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 11,215,873
Trainable params: 11,215,873
Non-trainable params: 0
_________________________________________________________________
```

This CNN model is tailored specifically to efficiently classify images of bees and wasps, leveraging its layered structure to learn intricate features and make accurate predictions.

## Data Generation and Model Training

To effectively train the image classification model, the dataset is prepared and augmented using `ImageDataGenerator` in TensorFlow. This section outlines the steps involved in generating the input data and configuring the training process.

### Data Generation

- **ImageDataGenerator:** Images are processed and augmented using `ImageDataGenerator` with rescaling (1./255) as the primary preprocessing step. No additional pre-processing is required for the images.
- **Class Mode Parameter:** For binary classification problems like distinguishing between bees and wasps, the `class_mode` parameter should be set to 'binary' when reading data from the train/test directories.
- **Batch Size:** Both training and test sets utilize a batch size of 20 for efficient processing.
- **Shuffling Data:** The datasets are shuffled (shuffle=True) to randomize the order of the data points during training and testing.

### Dataset Information
- **Training Dataset:** The training dataset comprises 3,678 images distributed among two classes: 'bee' and 'wasp'.
- **Class Indices:** The `class_indices` attribute of the training dataset specifies the mapping of classes to numerical indices ('bee': 0, 'wasp': 1), aiding in model interpretation and evaluation.
- **Test Dataset:** The test dataset consists of images from the './data/test' directory with a similar configuration as the training dataset, targeting a binary classification task with images of bees and wasps.
- **Efficient Training:** The batch-wise processing and shuffling of datasets ensure efficient utilization of resources during the training process, contributing to model convergence and accuracy.

### Model Training and Evaluation
The model was trained over 10 epochs using the `fit()` method. Throughout the epochs, both training and validation accuracies and losses were tracked. The model progressively improved its performance across epochs, as indicated by the increase in accuracy and decrease in loss for both training and validation datasets.

#### Training Progress
- **Training Accuracy Median:** The median training accuracy across all epochs was 0.80, demonstrating consistent improvement and learning by the model.
- **Training Loss Standard Deviation:** The standard deviation of the training loss across epochs was 0.091, indicating variations in the model's learning rate during different training phases.

#### Model Performance
- The initial epoch started with an accuracy of 0.542 and a loss of 0.685, both for training and validation datasets.
- Over subsequent epochs, the model showcased improvement, reaching a training accuracy of 0.816 and a validation accuracy of 0.740 by the final epoch.
- Losses reduced significantly, with training loss decreasing from 0.685 to 0.432, and validation loss showing a similar decreasing trend from 0.664 to 0.532 by the last epoch.

#### Training Visualization
The line plots demonstrate the training progress across epochs:

![Line plots of training and validation loss and accuracy](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-8/HW8_loss_acc_1.png)

- **Loss Plot:** Shows a consistent decrease in both training and validation losses, indicating improved model convergence.
- **Accuracy Plot:** Demonstrates increasing accuracies for both training and validation datasets, indicating the model's learning capability and ability to generalize.

These insights into model training and evaluation showcase its ability to learn from the data, steadily improving performance over epochs, and ultimately achieving a reasonable level of accuracy for the binary classification task of identifying bees and wasps in images. 

Adjustments and further fine-tuning of the model architecture or hyperparameters could potentially enhance its performance.

### Data Augmentation and Model Extension

#### Augmenting Training Data
- Augmentation parameters were added to the training data generator, expanding the dataset with increased diversity through rotation, shifting, and flipping.
- These augmentations contributed to a more robust training process, introducing variations to the existing images for improved model generalization.

#### Extended Model Training
- The model, already trained for 10 epochs, continued its training for another 10 epochs using the augmented dataset.
- The model exhibited promising performance, with an average test loss of approximately 0.48 across epochs 11-20, showcasing consistent improvement and effective learning.
- During the last 5 epochs (from 6 to 10), the model achieved an average test accuracy of about 0.78, indicating its ability to maintain a high level of accuracy on unseen data even after augmentation.

#### Training Visualization
The line plots illustrate the training progress across all epochs:

![Line plots of training and validation loss and accuracy](https://github.com/JasonDahl/mlzoomcamp-homework/blob/main/homework-8/HW8_loss_acc_2.png)

- **Loss Plot:** Displays a consistent decrease in both training and validation losses, indicating improved convergence even after augmenting the dataset.
- **Accuracy Plot:** Shows sustained increases in both training and validation accuracies, confirming the model's capability to learn from augmented data and generalize well.

The implementation of data augmentation further enhanced the model's learning and generalization abilities, demonstrating improved performance with increased dataset diversity and extended training.


## Conclusion
This project extensively explores CNNs for image classification tasks, utilizing TensorFlow and Keras. It covers essential aspects such as dataset preparation, model architecture, data augmentation, and iterative model training. The use of data augmentation significantly enhances the model's performance, showcasing its efficacy in deep learning tasks with image datasets. The application can be extended to various domains requiring image classification, demonstrating the versatility of deep learning techniques.
