# CitizenUAV Python

Implementation of my master thesis "Deep weed detection in agriculture - Adapting and improving deep-learning-based weed detection in agricultural high resolution UAV imagery using citizen science data"

## Abstract

Climate Change might be the greatest challenge of humanity in the 21st century.
The drivers of climate change have many dimensions, so there are many fields on
which humanity can fight it. One of these fields is human food supply. Almost all
food that is produced has its origins in plants, whether the plants are eaten directly
or are being further processed through animal bodies into meat or other animal
products. These plants are grown in the industry of agriculture, which itself has a
share in global greenhouse gas emissions of over 10% (Rolnick et al., 2019). That is,
because, to be easily manageable on a large scale, mono-cultures and chemical
pesticides and fertilizers are widely used. As a result large ecosystems in the areas
used as farmland are destroyed and thus a large amount of stored carbon is released
into the atmosphere. This calls for a solution to easily manage crop growing without
destroying the surrounding ecosystems. One such solution can be precision weeding,
in other words, automated weed detection, classification and removal if the
individual is expected to harm the crop. Precision weeding is part of the larger
research field called "precision agriculture". This thesis focuses on the detection and
classification of weed on large farmlands. For this task Machine Learning offers
great potential. Easily acquired UAV imagery can be fed to algorithms that detect
and classify certain plant species. While classical machine learning algorithms need
a lot of manual feature engineering, the field of Deep Learning offers models that are
able to learn by themselves what features are important (Kattenborn et al., 2021).
Deep learning models on the other hand have way more parameters than classical
models, and thus, need way more heterogeneous data, so that they can generalize
properly. A recent study by Soltani et al. (2022) has shown that it is possible to use
citizen science data from an open access database for training such a model and
apply it to side specific data of a natural ecosystem. The aim of this thesis is to
transfer that approach to an agricultural setting and use an ensemble of state of the
art deep learning techniques in order to improve the approach and adapt it to that
specific domain.

## Packages

### CitizenUAV

This package contains the implementation of all the model architectures, data pipelines and further procedures used for my thesis. 

**data.py:** Online dataset representations and pipline classes including specific complex I/O-routines.
**io.py:** Small general I/O-utilities.
**losses.py:** Custom loss functions used for style-transfer-based visualization techniques.
**math.py:** Small general mathematic utilities.
**models.py:** Model architecture configurations and model-specific training routines.
**processes.py:** Collection of procedures, e. g. augmentation or full dataset prediction. 
**transforms.py:** Cutsom data transformation techniques for augmentation.

### Moganet

This package includes an [implementation of the MogaNet architecture](https://github.com/Westlake-AI/MogaNet).

### SampleSelector

This package is contains a small PyQT app for manually filtering an image dataset based on visual appearance.
