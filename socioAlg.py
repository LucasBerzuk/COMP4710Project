# Socioeconomic algorithm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

RANDOM_STATE = 4710

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

# Merge data into one dataset on Name, eliminate other CrimeData columns because pointless
merged_data = pd.merge(data_computed, CrimeData[['Name', 'Crime Category']], on='Name', how='inner')

name_counts = merged_data['Name'].value_counts()
pd.set_option('display.max_rows', None)
print(name_counts)
print(len(name_counts))

# Set features
features = [
  'Pop. Density (per km^2)', 'Ownership Ratio', 'No Certification Ratio', 'Minority Ratio', 'Avg. Income', 'Employed Ratio'
  # 'High School Diploma Ratio', 'Post Secondary Ratio', 'Employed Ratio',
  # 'Avg. Income', 'Household Income Ratio', 'Ownership Ratio', 'Pop. Density (per km^2)', 
  # 'Indigenous Ratio', 'Minority Ratio', 'Immigrant Ratio', 
  # 'No Certification Ratio', 'LICO Ratio', 'Male Ratio', 'Female Ratio'
]

# Map the 'Crime Category' to 'Violent Crime' and 'Non-Violent Crime'
merged_data['Binary_Crime_Category'] = merged_data['Crime Category'].apply(
   lambda x: 'Violent Crime' if x in ['Violent Crime', 'Other Violent Crime'] else 'Non-Violent Crime')

# Now set the target variable to the new binary category
target = 'Binary_Crime_Category'

# Split the data into features and target
X = merged_data[features] 
y = merged_data[target]  

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

# # --------------------------------------MLP-NN-------------------------------------
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

# -----------------------------------------RF------------------------------------
# from sklearn.ensemble import RandomForestClassifier

# RF_model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
# RF_model.fit(X_train, y_train)

# RF_pred = RF_model.predict(X_test)
# print("RF Result")
# print(classification_report(y_test, RF_pred, zero_division=0))

# from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve

# # Get predicted probabilities for both classes
# RF_probs = RF_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (Violent Crime)

# # Define a custom threshold
# threshold = 0.25  # Lower threshold to favor Violent Crimes

# # Generate predictions based on the threshold
# RF_custom_pred = (RF_probs >= threshold).astype(int)

# # Convert class labels for evaluation
# y_test_binary = (y_test == 'Violent Crime').astype(int)  # Convert labels to binary (0 for Non-Violent, 1 for Violent)

# # Evaluate the model
# print("Classification Report (Threshold = {:.2f})".format(threshold))
# print(classification_report(y_test_binary, RF_custom_pred, zero_division=0))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test_binary, RF_custom_pred)
# print("Confusion Matrix:")
# print(conf_matrix)

# # Plot Precision-Recall Curve
# precision, recall, _ = precision_recall_curve(y_test_binary, RF_probs)
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 6))
# plt.plot(recall, precision, marker='.', label="Random Forest")
# plt.title("Precision-Recall Curve")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend()
# plt.grid()
# plt.show()

# # Plot ROC Curve
# fpr, tpr, _ = roc_curve(y_test_binary, RF_probs)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, marker='.', label="Random Forest")
# plt.title("ROC Curve")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.grid()
# plt.show()


# # ---------------------------------------K-NN-----------------------------------------
# from sklearn.neighbors import KNeighborsClassifier

# KNN_model = KNeighborsClassifier(n_neighbors=5,  # Number of neighbors (k)
#                                  metric='minkowski',  # Distance metric (Minkowski is the default)
#                                  p=2,                 # p=2 corresponds to Euclidean distance
#                                  n_jobs=-1)           # Use all processors
# KNN_model.fit(X_train_smote, y_train_smote)

# KNN_pred = KNN_model.predict(X_test_scaled)
# print("KNN Result")
# print(classification_report(y_test, KNN_pred, zero_division=0))

# # --------------------------------------DT------------------------------------------
# from sklearn.tree import DecisionTreeClassifier

# dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE,
#                                   criterion='gini',   # Use Gini impurity for decision splits (default)
#                                   max_depth=5,        # Limit the depth of the tree to prevent overfitting
#                                   min_samples_split=10, # Minimum samples required to split a node
#                                   min_samples_leaf=5, class_weight='balanced')  # Minimum samples required to be at a leaf node
# dt_model.fit(X_train_smote, y_train_smote)

# dt_pred = dt_model.predict(X_test_scaled)
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