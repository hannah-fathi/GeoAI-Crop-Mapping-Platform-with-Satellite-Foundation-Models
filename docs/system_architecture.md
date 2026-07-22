# System Architecture
---

## 1. Architecture Overview

The proposed system follows an end-to-end layered GeoAI
architecture designed for agricultural intelligence,
satellite representation learning, and automated crop
mapping.

The architecture integrates geospatial data processing,
satellite foundation model representations, machine learning
classification, and GIS-based decision support into a unified
workflow.


The complete processing pipeline consists of five major layers:


1. User Interface Layer

2. GeoAI Processing Engine

3. Satellite Representation Learning Layer

4. Machine Learning Intelligence Layer

5. GIS Visualization and Output Layer



```
Agricultural Field Polygons
            |
            v
Geospatial Data Engineering
            |
            v
Satellite Foundation Representations
     +----------------+
     |                |
     v                v
 AlphaEarth       Galileo
     |                |
     +----------------+
            |
            v
Feature Engineering
            |
            v
Vegetation Intelligence (NDVI + Polygon-MLP)
            |
            v
Crop Classification (Crop-MLP)
            |
            v
GIS Agricultural Intelligence Products
```

---

# 2. Architectural Design Principles


The system was designed based on the following principles:

## 2.1 End-to-End Processing

The platform provides a complete workflow from raw
agricultural field boundaries to final crop intelligence
products.

The pipeline includes:

* spatial data ingestion
* feature extraction
* machine learning inference
* evaluation
* GIS visualization

---

## 2.2 Separation of Responsibilities

Although the current implementation is provided as a
single Python application, the software architecture
follows a modular conceptual design.

Each logical component has a dedicated responsibility:

* data management
* representation extraction
* model learning
* evaluation
* visualization

---

## 2.3 Research-Oriented Design

The architecture supports experimentation with different
satellite representation sources and machine learning
configurations.

Examples:

* AlphaEarth only

* Galileo only

* AlphaEarth + Galileo feature fusion

* Different embedding dimensions

* Different classification models

---

# 3. Software Components


## 3.1 Application Layer

### Responsibility

The Application Layer provides the interactive environment
for configuring and executing the GeoAI workflow.

### Main Functions

* User interaction

* Parameter configuration

* Workflow execution control

* Model training management

* Prediction execution

### Implemented Capabilities

The graphical interface provides dedicated modules for:

### Data Loading and Preview

Functions:

* Upload agricultural polygon files
* Preview spatial data
* Display polygons on interactive maps

### Embedding and MLP Configuration

Functions:

* Configure AlphaEarth parameters
* Configure Galileo inference
* Select feature combinations
* Train vegetation model

### Crop Classification Interface

Functions:

* Load labeled crop data
* Train crop classifier
* Generate prediction maps
* Display evaluation metrics

---

# 3.2 Data Processing Layer

## Responsibility

The Data Processing Layer manages all geospatial
preprocessing operations before machine learning.

## Main Operations

### Vector Data Loading

Input:

Agricultural polygon datasets

Supported format:

* ZIP vector packages

---

### Geometry Processing

Operations:

* Geometry validation
* Area filtering
* Boundary simplification
* Coordinate system management

---

### Polygon Identification

Each agricultural field is assigned a unique identifier:

```
poly_id
```

This identifier enables:

* feature tracking
* model prediction mapping
* GIS output generation

---

# 3.3 Satellite Representation Layer

## Responsibility

This layer extracts meaningful satellite representations
from Earth observation data.

The system supports two representation sources:

# AlphaEarth Representation

AlphaEarth provides learned satellite embeddings that
capture temporal and spectral characteristics of
agricultural regions.

Supported feature configurations:

## Mean Embedding

Dimension:

64

## Statistical Embedding

Features:

* Mean
* Median
* First Quartile
* Third Quartile

Dimension:

256

## Combined Representation

AlphaEarth 64-dimensional embedding combined with
statistical features:

Dimension:

320

---

# Galileo Representation

Galileo provides deep satellite feature representations
generated from Sentinel-2 observations.

The system supports:

* Galileo standalone features

* AlphaEarth + Galileo feature fusion

---

# 3.4 Machine Learning Intelligence Layer

## Responsibility

This layer performs learning, prediction, and evaluation.

The architecture contains two machine learning models:

# Polygon-MLP Vegetation Classification Model

## Objective

Classify agricultural polygons into:

* Vegetation

* Non-Vegetation

* Unknown

## Input

Satellite embedding features

## Output

Vegetation mask

---

# Crop-MLP Classification Model

## Objective

Perform multi-class crop classification.

## Input

Vegetation-positive polygon embeddings

## Output

Five crop classes

## Training Strategy

The model uses:

* supervised learning

* feature normalization

* stratified train-test split

---

# 3.5 Evaluation Layer

## Responsibility

The evaluation module measures model performance
and generates quantitative reports.

Supported metrics:

* Accuracy

* Precision

* Recall

* F1-score

* Macro F1-score

Evaluation outputs:

* Classification reports

* Performance tables

* Prediction summaries

---

# 3.6 Visualization and GIS Output Layer

## Responsibility

This layer transforms model predictions into interpretable
geospatial products.

Generated outputs:

## Spatial Products

* GeoJSON crop maps

* GeoPackage (GPKG) layers

## Visualization Products

* Crop distribution maps

* Prediction visualization

* Model performance charts

---

# 4. Complete Data Flow

The complete system workflow can be summarized as:

```
Agricultural Polygon Data

        ↓

Geospatial Preprocessing

        ↓

Satellite Representation Extraction

        ↓

AlphaEarth / Galileo Embedding Generation

        ↓

Feature Engineering

        ↓

NDVI Vegetation Analysis

        ↓

Polygon-MLP Vegetation Classification

        ↓

Crop-MLP Multi-Class Classification

        ↓

GIS Map Generation

        ↓

Agricultural Intelligence Products
```

---

# 5. Implementation Note

The current repository provides a research-oriented
implementation of the complete pipeline in Python.

The implementation integrates:

* Geospatial computing
* Satellite foundation representations
* Machine learning
* GIS visualization
* Interactive application development

The architecture is designed to support future
extensions including deep learning models,
foundation model adaptation, and large-scale
agricultural monitoring systems.
