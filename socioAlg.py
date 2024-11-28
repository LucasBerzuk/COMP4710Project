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
# positive_indicators = [
#     'Post Secondary Certificate, Diploma or Degree', 'Employed', 'Avg. Income',
#     'Median Income', 'Avg. Household Income', 'Dwelling-Owned'
# ]
# negative_indicators = [
#     'No Certificate, Diploma or Degree', 'Unemployed', 'LICO-AT', 'Dwelling-Rented',
#     'Avg. Person per House'
# ]

all_indicators = [
  'Pop.', 'Pop. Density (per km^2)', 'Male Pop.', 'Female Pop.', 'Indigenous Identity',
  'Visible Minorities', 'Immigrants (place of birth)', 'No Certificate, Diploma or Degree',
  'High School Diploma or Equiv.', 'Post Secondary Certificate, Diploma or Degree',
  'Employed', 'Unemployed', 'Avg. Income', 'LICO-AT', 'Avg. Person per House', 
  'Avg. Household Income', 'Dwelling-Owned', 'Dwelling-Rented'
]

# Ensure all relevant columns are numeric
CensusData[all_indicators] = CensusData[all_indicators].apply(pd.to_numeric, errors='coerce')

# Initialize scalers
scaler = MinMaxScaler()

# Function to compute ratios with a check for zero in the denominator or numerator
def safe_ratio(numerator, denominator):
    return 0 if denominator == 0 or numerator == 0 else numerator / denominator

# Normalize positive indicators
# List of positives
# # High School Diploma/Pop: Higher ratio is better
# CensusData['High School Diploma Ratio'] = CensusData['High School Diploma or Equiv.'] / CensusData['Pop.']
# # Post Secondary Cert/Pop: Higher ratio is better
# CensusData['Post Secondary Ratio'] = CensusData['Post Secondary Certificate, Diploma or Degree'] / CensusData['Pop.']
# # Employed/(Employed+Unemployed): Higher ratio is better
# CensusData['Employed Ratio'] = CensusData['Employed'] / (CensusData['Employed'] + CensusData['Unemployed'])
# # Avg Income: Higher = better
# CensusData['Avg. Income'] = CensusData['Avg. Income']
# # Household Income/Avg. PPL per House: Higher num is better
# CensusData['Household Income Ratio'] = CensusData['Avg. Household Income'] / CensusData['Avg. Person per House']
# # Owned/Rented: Higher = better
# CensusData['Ownership Ratio'] =  CensusData['Dwelling-Owned'] / CensusData['Dwelling-Rented']

# High School Diploma/Pop: Higher ratio is better
CensusData['High School Diploma Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['High School Diploma or Equiv.'], row['Pop.']), axis=1)
# Post Secondary Cert/Pop: Higher ratio is better
CensusData['Post Secondary Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Post Secondary Certificate, Diploma or Degree'], row['Pop.']), axis=1)
# Employed/(Employed + Unemployed): Higher ratio is better
CensusData['Employed Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Employed'], row['Employed'] + row['Unemployed']), axis=1)
# Avg Income: Higher = better
CensusData['Avg. Income'] = CensusData['Avg. Income']
# Household Income/Avg. PPL per House: Higher num is better
CensusData['Household Income Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Avg. Household Income'], row['Avg. Person per House']), axis=1)
# Owned/Rented: Higher is better
CensusData['Ownership Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Dwelling-Owned'], row['Dwelling-Rented']), axis=1)

positive_indicators = [
  'High School Diploma Ratio', 'Post Secondary Ratio', 'Employed Ratio',
  'Avg. Income', 'Household Income Ratio', 'Ownership Ratio'
]

data_normalized = CensusData.copy()
data_normalized[positive_indicators] = scaler.fit_transform(CensusData[positive_indicators])

# Normalize negative indicators (invert scale)
# List of negatives
# Pop. density: Higher is worse
# CensusData['Pop. Density (per km^2)'] = CensusData['Pop. Density (per km^2)']
# # Indigenous/Pop: Higher is worse
# CensusData['Indigenous Ratio'] =  CensusData['Indigenous Identity'] / CensusData['Pop.']
# # Minorities/Pop: Higher is worse
# CensusData['Minority Ratio'] =  CensusData['Visible Minorities'] / CensusData['Pop.']
# # Immigrants/Pop: Higehr is worse
# CensusData['Immigrant Ratio'] =  CensusData['Immigrants (place of birth)'] / CensusData['Pop.']
# # No certification/Pop: Higher is worse
# CensusData['No Certification Ratio'] =  CensusData['No Certificate, Diploma or Degree'] / CensusData['Pop.']
# # LICO/Pop: Higher is worse
# CensusData['LICO Ratio'] =  CensusData['LICO-AT'] / CensusData['Pop.']

# Pop. density: Higher is worse
CensusData['Pop. Density (per km^2)'] = CensusData['Pop. Density (per km^2)']
# Indigenous/Pop: Higher is worse
CensusData['Indigenous Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Indigenous Identity'], row['Pop.']), axis=1)
# Minorities/Pop: Higher is worse
CensusData['Minority Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Visible Minorities'], row['Pop.']), axis=1)
# Immigrants/Pop: Higehr is worse
CensusData['Immigrant Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Immigrants (place of birth)'], row['Pop.']), axis=1)
# No certification/Pop: Higher is worse
CensusData['No Certification Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['No Certificate, Diploma or Degree'], row['Pop.']), axis=1)
# LICO/Pop: Higher is worse
CensusData['LICO Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['LICO-AT'], row['Pop.']), axis=1)

negative_indicators = [
  'Pop. Density (per km^2)', 'Indigenous Ratio', 'Minority Ratio',
  'Immigrant Ratio', 'No Certification Ratio', 'LICO Ratio'
]

data_normalized[negative_indicators] = 1 - scaler.fit_transform(CensusData[negative_indicators])


# Normalize male/female pop because its worse either way from 1 the further it goes
# Calculate the male-to-female ratio
CensusData['Gender Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Male Pop.'], row['Female Pop.']), axis=1)

# Calculate the deviation from the ideal ratio (1)
CensusData['Gender Deviation'] = abs(1 - CensusData['Gender Ratio'])

# Normalize the deviation (1 is ideal, further away is worse)
max_deviation = CensusData['Gender Deviation'].max()
data_normalized['Normalized Gender Score'] = 1 - (CensusData['Gender Deviation'] / max_deviation)



# Combine normalized values into a socio-economic score
weights = {
  'High School Diploma Ratio': 0.1,
  'Post Secondary Ratio': 0.15,
  'Employed Ratio': 0.15,
  'Avg. Income': 0.08,
  'Household Income Ratio': 0.08,
  'Ownership Ratio': 0.05,
  'Pop. Density (per km^2)': 0.02,
  'Indigenous Ratio': 0.01,
  'Minority Ratio': 0.01,
  'Immigrant Ratio': 0.01,
  'No Certification Ratio': 0.15,
  'LICO Ratio': 0.15,
  'Normalized Gender Score': 0.04

    # 'Post Secondary Certificate, Diploma or Degree': 0.2,
    # 'Employed': 0.2,
    # 'Avg. Income': 0.15,
    # 'Median Income': 0.15,
    # 'Avg. Household Income': 0.15,
    # 'Dwelling-Owned': 0.1,
    # 'No Certificate, Diploma or Degree': 0.05,
    # 'Unemployed': 0.05,
    # 'LICO-AT': 0.1,
    # 'Dwelling-Rented': 0.05,
    # 'Avg. Person per House': 0.05
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