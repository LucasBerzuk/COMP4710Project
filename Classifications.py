# Socioeconomic algorithm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

RANDOM_STATE = 184

# create pickel files to load data quickly
try:
  CensusData = joblib.load("CensusNeighbourhoods.pkl")
  CrimeData = joblib.load("CrimeByNeighbourhood.pkl")
  print("Loaded from cache\n")

except FileNotFoundError:
  CensusData = pd.read_csv("CensusNeighbourhoods.csv")
  CrimeData = pd.read_csv("CrimeByNeighbourhood.csv")
  joblib.dump(CensusData, "CensusNeighbourhoods.pkl")
  joblib.dump(CrimeData, "CrimeByNeighbourhood.pkl")
  print("Loaded from CSV and cached\n")

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

# List of Computed values based on population etc of a neighbourhood
CensusData['High School Diploma Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['High School Diploma or Equiv.'], row['Pop.']), axis=1)

CensusData['Post Secondary Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Post Secondary Certificate, Diploma or Degree'], row['Pop.']), axis=1)

CensusData['Employed Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Employed'], row['Employed'] + row['Unemployed']), axis=1)

CensusData['Avg. Income'] = CensusData['Avg. Income']

CensusData['Household Income Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Avg. Household Income'], row['Avg. Person per House']), axis=1)

CensusData['Ownership Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Dwelling-Owned'], row['Dwelling-Owned'] + row['Dwelling-Rented']), axis=1)

CensusData['Pop. Density (per km^2)'] = CensusData['Pop. Density (per km^2)']

CensusData['Indigenous Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Indigenous Identity'], row['Pop.']), axis=1)

CensusData['Minority Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Visible Minorities'], row['Pop.']), axis=1)

CensusData['Immigrant Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Immigrants (place of birth)'], row['Pop.']), axis=1)

CensusData['No Certification Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['No Certificate, Diploma or Degree'], row['Pop.']), axis=1)

CensusData['LICO Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['LICO-AT'], row['Pop.']), axis=1)

CensusData['Male Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Male Pop.'], row['Pop.']), axis=1)

CensusData['Female Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['Female Pop.'], row['Pop.']), axis=1)

all_computed_indicators = [
  'High School Diploma Ratio', 'Post Secondary Ratio', 'Employed Ratio',
  'Avg. Income', 'Household Income Ratio', 'Ownership Ratio', 'Pop. Density (per km^2)', 
  'Indigenous Ratio', 'Minority Ratio', 'Immigrant Ratio', 
  'No Certification Ratio', 'LICO Ratio', 'Male Ratio', 'Female Ratio'
]

# Trim down to just the features we need
data_computed = pd.DataFrame({'Name': CensusData['Name']})
data_computed[all_computed_indicators] = CensusData[all_computed_indicators]

# ---------------------------
# Classification Testing
# ---------------------------

# Count crimes per neighborhood
crime_count = CrimeData.groupby('Name').size().reset_index(name='Crime Count')

# Merge crime count with the census data
data_computed = pd.merge(CensusData, crime_count, left_on='Name', right_on='Name', how='left')

# Handle missing values and preprocess the data
# Fill missing crime counts with 0 for neighborhoods with no reported crime
data_computed['Crime Count'] = data_computed['Crime Count'].fillna(0)

# Add a new column 'Crime Rate' based on the crime count
def categorize_crime_rate(count):
    if count <= 99:
        return 'Low'
    elif count <= 299:
        return 'Mid'
    else:
        return 'High'

# Apply the function to create the 'Crime Rate' column
data_computed['Crime Rate'] = data_computed['Crime Count'].apply(categorize_crime_rate)

# Set features
features = [
   'No Certification Ratio', 'Minority Ratio','Employed Ratio','Pop. Density (per km^2)','Avg. Income','Ownership Ratio'
#   'High School Diploma Ratio', 'Post Secondary Ratio', 'Employed Ratio',
#   'Avg. Income', 'Household Income Ratio', 'Ownership Ratio', 'Pop. Density (per km^2)', 
#   'Indigenous Ratio', 'Minority Ratio', 'Immigrant Ratio', 
#   'No Certification Ratio', 'LICO Ratio', 'Male Ratio', 'Female Ratio'
]

# Now set the target variable to the new binary category
target = 'Crime Rate'

# Split the data into features and target
X = data_computed[features] 
y = data_computed[target]  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = RANDOM_STATE)

# Ensure that both X_train and X_test are DataFrames to retain feature names
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

# ------------------------Scaling--------------------------------
from sklearn.preprocessing import StandardScaler, RobustScaler

# Instantiate the scaler
scaler = RobustScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)  # Calculate mean/std and scale
X_test_scaled = scaler.transform(X_test)       # Use the same mean/std to scale

# ----------------------------SMOTE----------------------------
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = RANDOM_STATE)  # Initialize SMOTE
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)  # Resample training data

# Ensure the resampled data is still in DataFrame format to avoid issues with model predictions
X_train_smote = pd.DataFrame(X_train_smote, columns=features)

# -----------------------------------------RF------------------------------------
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 100,
                                  criterion= 'gini',
                                  max_depth = None,
                                  min_samples_split = 2,
                                  min_samples_leaf = 1,
                                  min_weight_fraction_leaf = 0,
                                  max_features = "sqrt",
                                  max_leaf_nodes = None,
                                  min_impurity_decrease = 0,
                                  bootstrap = True,
                                  oob_score = False,
                                  random_state=RANDOM_STATE, 
                                  class_weight='balanced')
RF_model.fit(X_train_smote, y_train_smote)

RF_pred = RF_model.predict(X_test)
print("RF Result")
print(classification_report(y_test, RF_pred, zero_division=0))

# --------------------------------------MLP-NN-------------------------------------
# from sklearn.neural_network import MLPClassifier

# MLP_model = MLPClassifier(hidden_layer_sizes=(100,100),  # One hidden layer with 100 neurons
#                           solver='adam',             # Optimizer (Adam)
#                           activation='relu',         # ReLU activation function
#                           max_iter=1500,             # Number of iterations for training
#                           random_state=RANDOM_STATE, alpha = 0.0383) # Random state for reproducibility
# MLP_model.fit(X_train, y_train)

# MLP_pred = MLP_model.predict(X_test)
# print("MLP Result")
# print(classification_report(y_test, MLP_pred, zero_division=0))

# ---------------------------------------K-NN-----------------------------------------
# from sklearn.neighbors import KNeighborsClassifier

# KNN_model = KNeighborsClassifier(n_neighbors=5,  # Number of neighbors (k)
#                                  metric='minkowski',  # Distance metric (Minkowski is the default)
#                                  p=2,                 # p=2 corresponds to Euclidean distance
#                                  n_jobs=-1)           # Use all processors
# KNN_model.fit(X_train, y_train)

# KNN_pred = KNN_model.predict(X_test)
# print("KNN Result")
# print(classification_report(y_test, KNN_pred, zero_division=0))

# --------------------------------------DT------------------------------------------
# from sklearn.tree import DecisionTreeClassifier

# dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE,
#                                   criterion='gini',   # Use Gini impurity for decision splits (default)
#                                   max_depth=5,        # Limit the depth of the tree to prevent overfitting
#                                   min_samples_split=10, # Minimum samples required to split a node
#                                   min_samples_leaf=5, class_weight='balanced')  # Minimum samples required to be at a leaf node
# dt_model.fit(X_train, y_train)

# dt_pred = dt_model.predict(X_test)
# print("DT Result")
# print(classification_report(y_test, dt_pred, zero_division=0))

# # ------------------------------------------GBDT----------------------------------------
# from sklearn.ensemble import GradientBoostingClassifier

# gbdt_model = GradientBoostingClassifier(n_estimators=100,    # Number of trees in the ensemble
#                                         learning_rate=0.1,   # Learning rate to shrink contribution of each tree
#                                         max_depth=5,         # Maximum depth of each individual tree
#                                         random_state=RANDOM_STATE)
# gbdt_model.fit(X_train_smote, y_train_smote)

# gbdt_pred = gbdt_model.predict(X_test_scaled)
# print("GBDT Result")
# print(classification_report(y_test, gbdt_pred, zero_division=0))

# # --------------------------------------------ET-------------------------------------------
# from sklearn.ensemble import ExtraTreesClassifier

# et_model = ExtraTreesClassifier(n_estimators=100,    # Number of trees in the ensemble
#                                 max_depth=5,         # Maximum depth of each individual tree
#                                 random_state=RANDOM_STATE,
#                                 n_jobs=-1, class_weight='balanced')           # Use all cores for faster computation
# et_model.fit(X_train_scaled, y_train)

# et_pred = et_model.predict(X_test_scaled)
# print("ET Result")
# print(classification_report(y_test, et_pred, zero_division=0))

# # --------------------------------------SVC---------------------------------------
# from sklearn.svm import SVC

# svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE, class_weight='balanced')
# svc_model.fit(X_train_smote, y_train_smote)  # Ensure target is binary

# svc_pred = svc_model.predict(X_test_scaled)
# print("SVC Result")
# print(classification_report(y_test, svc_pred, zero_division=0))