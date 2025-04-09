# OceanInsight: Argo Float Analysis Platform

## Overview
OceanInsight is a high-performance data processing and analysis platform designed for oceanographic research using Argo float data. This project processes vertical temperature profiles across ocean depths, enabling advanced climate studies and spatial analysis of ocean conditions over large geographical areas.

The platform leverages Apache Spark for scalable data processing and machine learning to analyze spatial distribution of ocean temperature at different depths, identify oceanic patterns, and model temperature relationships.

## Features
- **Data Processing & Cleaning:** Handles large-scale Argo float datasets with quality control
- **Statistical Analysis:** Comprehensive analysis of temperature distribution across depth ranges
- **Machine Learning Models:**
  - Classification models to categorize warm vs. cold water masses
  - Regression models to predict temperature based on location and pressure
  - Clustering to identify distinct water masses
- **Visualizations:** Geographic distribution maps, temperature-pressure relationships, time series analysis
- **Optimized Performance:** Configured for efficient processing on standard hardware

## Technical Stack
- PySpark for distributed data processing
- PySpark ML for machine learning models
- Pandas for data manipulation
- Matplotlib and Seaborn for visualization
- Scikit-learn compatible ML pipeline

## Installation

### Prerequisites
- Apache Spark 3.x
- Python 3.7+
- Java 8+

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/hsn07pk/oceaninsight.git
   cd oceaninsight
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download sample Argo float data or use your own dataset in CSV format.

## Usage

1. Place your Argo float data in a file named `data.csv` in the project directory.

2. Run the analysis script:
   ```
   python argo_analysis.py
   ```

3. View generated visualizations in the `visualizations` directory.

## Data Requirements
The script expects a CSV file with the following columns:
- `time`: Timestamp of measurement
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `pres_adjusted`: Adjusted pressure measurement (dbar)
- `temp_adjusted`: Adjusted temperature measurement (°C)
- `pres_adjusted_qc`: Quality flag for pressure (optional)
- `temp_adjusted_qc`: Quality flag for temperature (optional)

## Model Performance
The platform builds and evaluates multiple models:
- Classification models (Logistic Regression, Random Forest) for predicting warm/cold water masses
- Regression models (Linear Regression, Random Forest) for temperature prediction
- K-means clustering for identifying distinct water masses

## Example Results
- Identification of distinct water masses based on temperature and pressure profiles
- Spatial distribution maps of ocean temperatures
- Temperature variation with depth analysis
- Time series analysis of temperature changes

## Future Enhancements
- Integration with real-time Argo data sources
- Enhanced 3D visualization capabilities
- Deep learning models for improved prediction accuracy
- Support for additional oceanographic parameters (salinity, oxygen, etc.)

## Citation
If you use this code in your research, please cite:
```
Author Name. (2025). OceanInsight: Argo Float Analysis Platform. GitHub repository. https://github.com/hsn07pk/oceaninsight
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The Argo Program for providing freely available oceanographic data
- The Apache Spark community for their excellent distributed computing framework