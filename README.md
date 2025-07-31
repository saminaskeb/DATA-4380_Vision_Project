![](UTA-DataScience-Logo.png)

# Project Title

* **One Sentence Summary**: This repository implements and compares transfer learning models for classifying race from facial images using the UTKFace dataset. 

## Overview

* **Definition of the task / challenge**:
The task is to classify human race categories from facial image data, using the UTKFace dataset. Each image is labeled with age, gender, and race, and we focus solely on race prediction as a multiclass classification task with 5 classes.

* **Your approach**:
We treat this as an image classification problem. Pretrained convolutional neural networks (CNNs) — MobileNetV2, ResNet50, and EfficientNetB0 — were used via transfer learning. We froze the base layers initially, trained a custom classifier head, then later fine-tuned upper base layers for improved performance.

* **Summary of the performance achieved**:
MobileNetV2 achieved ~45% accuracy after fine-tuning on a balanced subset of 400 images per class. Other models like ResNet50 and EfficientNetB0 performed less optimally on this specific subset.

## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Input: UTKFace dataset — JPEG facial images, around 200x200 pixels
  * Output: One of 5 race classes (0 to 4)
  * Size: Original dataset contains ~11000 images
  * Split used: 400 images per class (total ~1600), split into 80% train / 20% validation

* Preprocessing / Clean-up
  * Filenames were parsed to extract race labels (format: age_gender_race_date.jpg)
  * Removed corrupted or unreadable images
  * Created directory structure for Keras flow: /race_id/image.jpg
  * Applied normalization (rescale=1./255) and basic augmentations for training

* Data Visualization
Plotted random samples from each class

Created class distribution bar charts

(Optional: Insert matplotlib figures here if desired)

### Problem Formulation

* Input / Output

  * Input: RGB facial images resized to 224x224
  * Output: Integer label for race class (0 to 4)

* Models Used
  * MobileNetV2 (baseline + fine-tuning)
  * ResNet50
  * EfficientNetB0

### Training

* Environment: Trained in Google Colab with GPU enabled
* Time: ~5 minutes per model for frozen base, ~10–15 minutes for fine-tuned
* Early stopping used to avoid overfitting
* Difficulties: Low accuracy initially with small sample size, resolved by increasing sample to 300/class and tuning augmentations
* Training curves were plotted for all models
(Insert training/validation accuracy/loss plots here if available)

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* MobileNetV2 consistently outperformed the others on this dataset
* Simpler or more efficient models seem better suited to this relatively small, imbalanced dataset
* Transfer learning is effective, but heavily influenced by data quantity and quality

### Future Work

* Expand dataset to use full UTKFace or synthetic augmentation
* Explore multi-label prediction (e.g., race + gender)

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







