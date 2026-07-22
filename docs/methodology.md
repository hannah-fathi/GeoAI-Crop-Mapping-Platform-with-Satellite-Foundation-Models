# Methodology


# 1. Overview


The proposed methodology consists of five major stages:


1. Geospatial preprocessing

2. Satellite embedding extraction

3. Vegetation detection

4. Crop classification

5. GIS output generation



---

# 2. Geospatial Data Processing


Input:

Agricultural field polygons


Processing:


## Polygon Validation

Ensuring valid geographic geometries.


## Area Filtering

Removing polygons below predefined area thresholds.


## Geometry Simplification

Reducing boundary complexity while preserving
spatial characteristics.


## Polygon Identification

Assigning unique identifiers:

poly_id



---

# 3. Satellite Representation Extraction


Two satellite representation sources are considered.


## 3.1 AlphaEarth Embeddings


AlphaEarth provides satellite-derived embeddings
representing temporal-spectral characteristics.


Feature configurations:


### Mean Embedding

Dimension:

64


### Statistical Embedding

Based on:

- Mean
- Median
- Q1
- Q3


Dimension:

256



---

# 3.2 Galileo Representation


Galileo provides learned satellite representations
using a deep neural feature extractor.


The framework supports:

- standalone Galileo features
- feature fusion with AlphaEarth



---

# 4. Vegetation Intelligence Module


Vegetation status is estimated using NDVI analysis.


NDVI provides a vegetation indicator:

Vegetation

Non-Vegetation


The generated labels are used for Polygon-MLP training.



---

# 5. Polygon-Level Classification


A Multi-Layer Perceptron model is trained
for vegetation classification.


Input:

Satellite embeddings


Output:

Vegetation categories



---

# 6. Crop Classification


Only vegetation-positive polygons are processed.


A second MLP model performs:

Five-class crop classification


Input:

Embedding feature matrix


Output:

Crop categories



---

# 7. Output Generation


Final outputs:


- GeoJSON crop maps
- GPKG spatial products
- Evaluation metrics
- Visualization maps
