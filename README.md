# 🚨🤖 ddos-detector
Semester project for Advanced Data Analysis Methods Laboratory

Team members: 

- Laszlo Patrik Abrok (JPWF8N)
- Janos Hudak (CP4EIQ)

## ℹ️ Project description

This is our semester project for the [VITMMB10 Advanced Data Analysis Methods Laboratory](https://portal.vik.bme.hu/kepzes/targyak/VITMMB10/en/) course. It leverages real-life network data collected by [SmartComLab](https://smartcomlab.tmit.bme.hu/) to classify DDoS attacks with high precision.

## 📊 Data 

The dataset is currently under revision for publication and therefore we considered it confidential. It can not be found in the repository, but we described properly in each milestone (and each project, or notebook) what is needed exactly to reproduce our work.

We also included a script in the root folder, `csv_to_pq.py`, that is capable of the required parquet conversion. The usage is described there.

## 📁 Files

As per our last milestone meeting, we organized our work into six folders to show each and every improvement. Throughout the semester, we experimented with many things: sometimes we needed robust projects, sometimes smaller, more flexible notebooks. The contents of the milestone folders contain our finished (and retrospectively refactored) code.

### M1

At the start of the semester we began the data preprocessing with a full, dockerized python (poetry) project. Though we did not see the future challenges, we tried prepare with the most robust solution we could, and included all the possible dependencies.

The most important file is `ddos_data_preprocessor.py`, which includes all the preprocessing functions that we later used.

### M2 

This was the first milestone, where a full python project was more of an overhead, than a help. We switched to Google Colaboratory, and used the dataset created by our M1 project, converted to parquet by our script. 

The most important file is `visualization.ipynb`, which features all the generalized functions for our visualization experiments.

### M3

Feature engineering and augmentation required many experiments, and we continued our work in Google Colaboratory because of this.

The `augmentation.ipynb` file contains our augmentation attempts with various methods, and `feature_engineering.ipynb` covers the feature engineering experiments.

### M4 

Testing machine learning solutions were also a trial-and-error heavy iteration, and we remained in notebooks for efficiency. 

The `tree_based.ipynb` file covers all attempts with tree-based models, and `neural_network.ipynb` contains the attempt with our feed-forward neural network.

### M5 

In this milestone, we focused on improvements, which included hyperparameter optimization. To make this more effective, we used our device resources for faster development. 

The `catboost/main.py` script contains the WandB integration with the sweeps. 

### M6

We continued our work in scripts executed on our devices. 

The `eval.py` file contains our generic evaluator functions, which we used for comparing the performances our models. We searched for optimal parameters in the scripts with the `sweep` prefix. Finally, `ensemble.py` covers our best-performing model in this semester.