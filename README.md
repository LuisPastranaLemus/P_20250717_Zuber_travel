# 🧭 Project Analysis

Data-Driven Insights for Zuber: Understanding Ride-Sharing Patterns in Chicago

---

## 🔍 Project Overview (P-202YMMDD_Name)

This project was developed as part of a data analysis initiative for Zuber, a new ride-sharing company launching operations in Chicago. The main objective is to identify key trends, passenger behavior, and external factors that influence ride demand.

Key questions:

- How do passengers typically use ride-sharing services?
- What patterns or trends exist in ride frequency, location, or time of day?
- How does weather affect the number of trips taken?
- What insights can we gain from competitor data to position Zuber effectively in the market?

Project Info explanation

This analysis is based on a combination of real-world datasets, including simulated ride logs and open-source weather data from the Chicago area. The project follows a structured approach:

1. Data Acquisition – Loading and cleaning the dataset(s) using Python and SQL.
2. Exploratory Data Analysis (EDA) – Visualizing trends, correlations, and usage patterns.
3. Hypothesis Testing – Evaluating the impact of external variables like weather conditions on ride frequency.

Insight Generation – Drawing actionable conclusions to support business strategy and operations.


> __Note__: All data used in this project has been anonymized or sourced from publicly available datasets. The focus is on demonstrating analytical methodology and business application rather than building a production-ready solution.

---

## 🧮 Data Dictionary

This project has a database of information on taxi rides in Chicago with 4 different tables.

- `neighborhoods` (data about the city's neighborhoods)
    `name`: neighborhood name
    `neighborhood_id`: neighborhood code

- `cabs` (data about taxis)
    `cab_id`: vehicle code
    `vehicle_id`: vehicle's technical ID
    `company_name`: the company that owns the vehicle

- `trips` (data about trips)
    `trip_id`: trip code
    `cab_id`: code of the vehicle operating the trip
    `start_ts`: trip start date and time (rounded to the nearest hour)
    `end_ts`: trip end date and time (rounded to the nearest hour)
    `duration_seconds`: trip duration in seconds
    `distance_miles`: trip distance in miles
    `pickup_location_id`: pickup neighborhood code
    `dropoff_location_id`: dropoff neighborhood code

- `weather_records` (weather data)
    `record_id`: weather record code
    `ts`: Date and time of recording (rounded to the nearest hour)
    `temperature`: Temperature when the recording was taken
    `description`: Brief description of the weather conditions, e.g., "light rain" or "partly cloudy"

---

## 📚 Guided Foundations (Historical Context)

The notebook `00-guided-analysis_foundations.ipynb` reflects an early stage of my data analysis learning journey, guided by TripleTen. It includes data cleaning, basic EDA, and early feature exploration, serving as a foundational block before implementing the improved structure and methodology found in the main analysis.

---

## 📂 Project Structure

```bash
├── data/
│   ├── raw/              # Original dataset(s) in CSV format
│   ├── interim/          # Intermediate cleaned versions
│   └── processed/        # Final, ready-to-analyze dataset
│
├── notebooks/
│   ├── 00-guided-analysis_foundations.ipynb     ← Initial guided project (TripleTen)
│   ├── 01_cleaning.ipynb                        ← Custom cleaning 
│   ├── 02_feature_engineering.ipynb             ← Custom feature engineering
│   ├── 03_eda_and_insights.ipynb                ← Exploratory Data Analysis & visual storytelling
│   └── 04-sda_hypotheses.ipynb                  ← Business insights and hypothesis testing
│
├── src/
│   ├── init.py              # Initialization for reusable functions
│   ├── data_cleaning.py     # Data cleaning and preprocessing functions
│   ├── data_loader.py       # Loader for raw datasets
│   ├── eda.py               # Exploratory data analysis functions
│   ├── features.py          # Creation and transformation functions for new variables to support modeling and EDA
│   └── utils.py             # General utility functions for reusable helpers
│
├── outputs/
│   └── figures/          # Generated plots and visuals
│
├── requirements/
│   └── requirements.txt      # Required Python packages
│
├── .gitignore            # Files and folders to be ignored by Git
└── README.md             # This file
```
---

🛠️ Tools & Libraries

- Python 3.11
- os, pathlib, sys, pandas, NumPy, Matplotlib, seaborn, IPython.display, scipy.stats 
- Jupyter Notebook
- Postgre
- Git & GitHub for version control

---

## 📌 Notes

This project is part of a personal learning portfolio focused on developing strong skills in data analysis, statistical thinking, and communication of insights. Constructive feedback is welcome.

---

## 👤 Author   
##### Luis Sergio Pastrana Lemus   
##### Engineer pivoting into Data Science | Passionate about insights, structure, and solving real-world problems with data.   
##### [GitHub Profile](https://github.com/LuisPastranaLemus)   
##### 📍 Querétaro, México     
##### 📧 Contact: luis.pastrana.lemus@engineer.com   
---

