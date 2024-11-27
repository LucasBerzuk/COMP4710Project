# Socioeconomic algorithm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# create pickel files to load data quickly
try:
  CensusData = joblib.load("CensusNeighbourhoods.pkl")
  print("Loaded from cache\n")

except FileNotFoundError:
  CensusData = pd.read_csv("CensusNeighbourhoods.csv")
  joblib.dump(CensusData, "CensusNeighbourhoods.pkl")
  print("Loaded from CSV and cached\n")


# Columns categorized as positive and negative indicators
positive_indicators = [
    'Post Secondary Certificate, Diploma or Degree', 'Employed', 'Avg. Income',
    'Median Income', 'Avg. Household Income', 'Dwelling-Owned'
]
negative_indicators = [
    'No Certificate, Diploma or Degree', 'Unemployed', 'LICO-AT', 'Dwelling-Rented',
    'Avg. Person per House'
]

# Ensure all relevant columns are numeric
CensusData[positive_indicators + negative_indicators] = CensusData[positive_indicators + negative_indicators].apply(pd.to_numeric, errors='coerce')

# Initialize scalers
scaler = MinMaxScaler()

# Normalize positive indicators
data_normalized = CensusData.copy()
data_normalized[positive_indicators] = scaler.fit_transform(CensusData[positive_indicators])

# Normalize negative indicators (invert scale)
data_normalized[negative_indicators] = 1 - scaler.fit_transform(CensusData[negative_indicators])

# Combine normalized values into a socio-economic score
weights = {
    'Post Secondary Certificate, Diploma or Degree': 0.2,
    'Employed': 0.2,
    'Avg. Income': 0.15,
    'Median Income': 0.15,
    'Avg. Household Income': 0.15,
    'Dwelling-Owned': 0.1,
    'No Certificate, Diploma or Degree': 0.05,
    'Unemployed': 0.05,
    'LICO-AT': 0.1,
    'Dwelling-Rented': 0.05,
    'Avg. Person per House': 0.05
}

# Calculate the socio-economic score
data_normalized['Socioeconomic Score'] = sum(
    data_normalized[column] * weight for column, weight in weights.items()
)

# Sort by score (highest score = best socio-economic status)
data_normalized_sorted = data_normalized.sort_values(by='Socioeconomic Score', ascending=False)

# Adjust pandas display settings to show all rows
pd.set_option('display.max_rows', None)

# Display the first few rows of the results
print(data_normalized_sorted[['Name', 'Socioeconomic Score']])
