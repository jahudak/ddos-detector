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

### M3

### M4 

### M5 

### M6