# Feature Engineering Results

Date: 2025-03-05 16:03:01

Total feature sets evaluated: 7

## Feature Set Rankings

| Feature Set    |      AUC |
|:---------------|---------:|
| polynomial     | 0.432871 |
| domain         | 0.430686 |
| dimensionality | 0.424063 |
| temporal       | 0.422254 |
| basic          | 0.422211 |
| statistical    | 0.420962 |

## Feature Set Details

### basic

**Description**: Basic features with minimal processing

**RMSE**: 0.42221

**Top 10 Features by Importance**:

| feature       |   importance |
|:--------------|-------------:|
| cloud         |    0.515544  |
| sunshine      |    0.0748223 |
| day           |    0.0568755 |
| windspeed     |    0.0556699 |
| dewpoint      |    0.0551157 |
| humidity      |    0.0494379 |
| pressure      |    0.0455227 |
| maxtemp       |    0.0442934 |
| mintemp       |    0.0402025 |
| winddirection |    0.0322443 |

### temporal

**Description**: Temporal and seasonal features

**RMSE**: 0.42225

**Top 10 Features by Importance**:

| feature   |   importance |
|:----------|-------------:|
| cloud     |    0.511359  |
| sunshine  |    0.0695458 |
| dewpoint  |    0.0494983 |
| windspeed |    0.0478646 |
| humidity  |    0.0440777 |
| maxtemp   |    0.0402773 |
| pressure  |    0.0377122 |
| mintemp   |    0.0355456 |
| day_sin   |    0.0348616 |
| day_cos   |    0.0295788 |

### statistical

**Description**: Statistical transformations of features

**RMSE**: 0.42096

**Top 10 Features by Importance**:

| feature          |   importance |
|:-----------------|-------------:|
| cloud_sqrt       |    0.105958  |
| cloud_z          |    0.0983655 |
| cloud            |    0.0928537 |
| cloud_scaled     |    0.0893323 |
| cloud_squared    |    0.0673241 |
| cloud_log        |    0.0622243 |
| sunshine_squared |    0.0165908 |
| sunshine_z       |    0.0153263 |
| sunshine_scaled  |    0.0142197 |
| dewpoint_z       |    0.014173  |

### polynomial

**Description**: Polynomial and interaction features

**RMSE**: 0.43287

**Top 10 Features by Importance**:

| feature               |   importance |
|:----------------------|-------------:|
| cloud_x_humidity      |    0.424554  |
| cloud                 |    0.0585209 |
| sunshine_div_dewpoint |    0.0480334 |
| cloud_x_dewpoint      |    0.0340556 |
| sunshine_div_cloud    |    0.0294121 |
| day                   |    0.0289586 |
| maxtemp               |    0.0202417 |
| pressure              |    0.0200749 |
| windspeed_x_dewpoint  |    0.020011  |
| sunshine_x_windspeed  |    0.019635  |

### dimensionality

**Description**: Dimensionality reduction techniques

**RMSE**: 0.42406

**Top 10 Features by Importance**:

| feature   |   importance |
|:----------|-------------:|
| pca_5_1   |    0.234886  |
| pca_10_1  |    0.1954    |
| cloud     |    0.165024  |
| pca_10_8  |    0.0346448 |
| pca_10_6  |    0.0315159 |
| sunshine  |    0.0301996 |
| pca_10_9  |    0.0290257 |
| pca_10_5  |    0.0278815 |
| pca_10_7  |    0.0265589 |
| maxtemp   |    0.0176548 |

### domain

**Description**: Domain-specific features for weather prediction

**RMSE**: 0.43069

**Top 10 Features by Importance**:

| feature             |   importance |
|:--------------------|-------------:|
| cloud               |    0.500312  |
| sunshine            |    0.0541083 |
| pressure_x_humidity |    0.0416601 |
| day                 |    0.0329229 |
| windspeed           |    0.0324654 |
| dewpoint            |    0.0293102 |
| pressure_ma3        |    0.0252533 |
| pressure_lag1       |    0.0232376 |
| maxtemp_x_humidity  |    0.0217342 |
| pressure            |    0.0212066 |

