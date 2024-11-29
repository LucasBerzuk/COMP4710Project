# Socioeconomic algorithm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
# High School Diploma/Pop: Higher ratio is better
CensusData['High School Diploma Ratio'] = CensusData.apply(
    lambda row: safe_ratio(row['High School Diploma or Equiv.'], row['Pop.']), axis=1)
# # Post Secondary Cert/Pop: Higher ratio is better

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
    lambda row: safe_ratio(row['Dwelling-Owned'], row['Dwelling-Owned'] + row['Dwelling-Rented']), axis=1)

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

# LICO/Pop: Higher is worse
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

# data_normalized = CensusData.copy()
data_computed = pd.DataFrame({'Name': CensusData['Name']})
data_computed[all_computed_indicators] = CensusData[all_computed_indicators]

# pd.set_option('display.max_columns', None)
# print(data_computed[data_computed['Name'] == 'AGASSIZ'])

# Normalize negative indicators (invert scale)
# List of negatives
# # Pop. density: Higher is worse
# CensusData['Pop. Density (per km^2)'] = CensusData['Pop. Density (per km^2)']

# # Indigenous/Pop: Higher is worse
# CensusData['Indigenous Ratio'] = CensusData.apply(
#     lambda row: safe_ratio(row['Indigenous Identity'], row['Pop.']), axis=1)

# # Minorities/Pop: Higher is worse
# CensusData['Minority Ratio'] = CensusData.apply(
#     lambda row: safe_ratio(row['Visible Minorities'], row['Pop.']), axis=1)

# # Immigrants/Pop: Higehr is worse
# CensusData['Immigrant Ratio'] = CensusData.apply(
#     lambda row: safe_ratio(row['Immigrants (place of birth)'], row['Pop.']), axis=1)

# # No certification/Pop: Higher is worse
# CensusData['No Certification Ratio'] = CensusData.apply(
#     lambda row: safe_ratio(row['No Certificate, Diploma or Degree'], row['Pop.']), axis=1)

# # LICO/Pop: Higher is worse
# CensusData['LICO Ratio'] = CensusData.apply(
#     lambda row: safe_ratio(row['LICO-AT'], row['Pop.']), axis=1)

# negative_indicators = [
#   'Pop. Density (per km^2)', 'Indigenous Ratio', 'Minority Ratio',
#   'Immigrant Ratio', 'No Certification Ratio', 'LICO Ratio'
# ]

# data_normalized[negative_indicators] = 1 - scaler.fit_transform(CensusData[negative_indicators])


# # Normalize male/female pop because its worse either way from 1 the further it goes
# # Calculate the male-to-female ratio
# CensusData['Gender Ratio'] = CensusData.apply(
#     lambda row: safe_ratio(row['Male Pop.'], row['Female Pop.']), axis=1)

# # Calculate the deviation from the ideal ratio (1)
# CensusData['Gender Deviation'] = abs(1 - CensusData['Gender Ratio'])

# # Normalize the deviation (1 is ideal, further away is worse)
# max_deviation = CensusData['Gender Deviation'].max()
# data_normalized['Normalized Gender Score'] = 1 - (CensusData['Gender Deviation'] / max_deviation)

# pd.set_option('display.max_columns', None)
# print(data_normalized[data_normalized['Name'] == 'AGASSIZ'])

# Combine normalized values into a socio-economic score
# weights = {
#   'Educated Ratio': 0.25,
#   'Employed Ratio': 0.15,
#   'Avg. Income': 0.08,
#   'Household Income Ratio': 0.08,
#   'Ownership Ratio': 0.05,
#   'Pop. Density (per km^2)': 0.02,
#   'Indigenous Ratio': 0.01,
#   'Minority Ratio': 0.01,
#   'Immigrant Ratio': 0.01,
#   'No Certification Ratio': 0.15,
#   'LICO Ratio': 0.15,
#   'Normalized Gender Score': 0.04
# }

# # Calculate the socio-economic score
# data_normalized['Socioeconomic Score'] = sum(
#     data_normalized[column] * weight for column, weight in weights.items()
# )

# # Sort by score (highest score = best socio-economic status)
# data_normalized_sorted = data_normalized.sort_values(by='Socioeconomic Score', ascending=False)

# # Adjust pandas display settings to show all rows
# pd.set_option('display.max_rows', None)

# # Display the first few rows of the results
# # print(data_normalized_sorted[['Name', 'Socioeconomic Score']])





# ---------------------------
# Classification Testing
# ---------------------------

# Merge data into one dataset on Name, eliminate other CrimeData columns because pointless
merged_data = pd.merge(data_computed, CrimeData[['Name', 'Crime Category']], on='Name', how='inner')

# Set features
features = [
  'Employed Ratio'
  #'High School Diploma Ratio', 'Post Secondary Ratio', 'Employed Ratio',
  #'Avg. Income', 'Household Income Ratio', 'Ownership Ratio', 'Pop. Density (per km^2)', 
  #'Indigenous Ratio', 'Minority Ratio', 'Immigrant Ratio', 
  #'No Certification Ratio', 'LICO Ratio', 'Male Ratio', 'Female Ratio'
]

# Map the 'Crime Category' to 'Violent Crime' and 'Non-Violent Crime'
merged_data['Binary_Crime_Category'] = merged_data['Crime Category'].apply(lambda x: 'Violent Crime' if x in ['Violent Crime', 'Other Violent Crime'] else 'Non-Violent Crime')

# Now set the target variable to the new binary category
target = 'Binary_Crime_Category'

# Split the data into features (X) and target (y)
X = merged_data[features]  # Features (socioeconomic and crime-related factors)
y = merged_data[target]    # Target (crime cat)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = RANDOM_STATE)

# Ensure that both X_train and X_test are DataFrames to retain feature names
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

# Scale data to try with and without scaling
from sklearn.preprocessing import StandardScaler, RobustScaler

# Instantiate the scaler
scaler = RobustScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)  # Calculate mean/std and scale
X_test_scaled = scaler.transform(X_test)       # Use the same mean/std to scale


# ----------------------------Balance the Data----------------------------

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = RANDOM_STATE)  # Initialize SMOTE
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)  # Resample training data

# Ensure the resampled data is still in DataFrame format to avoid issues with model predictions
X_train_smote = pd.DataFrame(X_train_smote, columns=features)

# from imblearn.over_sampling import RandomOverSampler

# ros = RandomOverSampler(random_state=RANDOM_STATE)
# X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
# X_train_ros = pd.DataFrame(X_train_ros, columns=features)


# --------------------------------------MLP-NN-------------------------------------
from sklearn.neural_network import MLPClassifier

MLP_model = MLPClassifier(hidden_layer_sizes=(100,100),  # One hidden layer with 100 neurons
                          solver='adam',             # Optimizer (Adam)
                          activation='relu',         # ReLU activation function
                          max_iter=1500,             # Number of iterations for training
                          random_state=RANDOM_STATE, alpha = 0.0383) # Random state for reproducibility
MLP_model.fit(X_train_smote, y_train_smote)

MLP_pred = MLP_model.predict(X_test_scaled)
print("MLP Result")
print(classification_report(y_test, MLP_pred, zero_division=0))
# SMOTE BETTER
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform, randint

# # Define the model
# mlp_model = MLPClassifier(random_state=42)

# # Define the hyperparameters to tune
# param_dist = {
#     'hidden_layer_sizes': [(50,), (100,), (100, 100), (200,)],
#     'activation': ['relu', 'tanh'],
#     'solver': ['adam', 'sgd'],
#     'alpha': uniform(0.0001, 0.1),  # Regularization term
#     'learning_rate': ['constant', 'invscaling', 'adaptive'],
#     'max_iter': [500, 1000, 1500],
#     'batch_size': ['auto', 128, 256]
# }

# # RandomizedSearchCV
# random_search = RandomizedSearchCV(estimator=mlp_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)

# # Fit randomized search
# random_search.fit(X_train_smote, y_train_smote)

# # Best parameters and score
# print("Best Parameters:", random_search.best_params_)
# print("Best Score:", random_search.best_score_)

# # Evaluate on the test set
# best_mlp_model = random_search.best_estimator_
# MLP_pred = best_mlp_model.predict(X_test_scaled)
# print(classification_report(y_test, MLP_pred))



# -----------------------------------------RF------------------------------------
# from sklearn.ensemble import RandomForestClassifier

# RF_model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
# RF_model.fit(X_train_smote, y_train_smote)

# RF_pred = RF_model.predict(X_test_scaled)
# print("RF Result")
# print(classification_report(y_test, RF_pred, zero_division=0))
#SMOTE BETTER

# ---------------------------------------K-NN-----------------------------------------
# from sklearn.neighbors import KNeighborsClassifier

# KNN_model = KNeighborsClassifier(n_neighbors=5,  # Number of neighbors (k)
#                                  metric='minkowski',  # Distance metric (Minkowski is the default)
#                                  p=2,                 # p=2 corresponds to Euclidean distance
#                                  n_jobs=-1)           # Use all processors
# KNN_model.fit(X_train_smote, y_train_smote)

# KNN_pred = KNN_model.predict(X_test_scaled)
# print("KNN Result")
# print(classification_report(y_test, KNN_pred, zero_division=0))
# SMOTE better


# --------------------------------------DT------------------------------------------
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
# SMOTE BETTER

# ------------------------------------------GBDT----------------------------------------
# from sklearn.ensemble import GradientBoostingClassifier

# gbdt_model = GradientBoostingClassifier(n_estimators=100,    # Number of trees in the ensemble
#                                         learning_rate=0.1,   # Learning rate to shrink contribution of each tree
#                                         max_depth=5,         # Maximum depth of each individual tree
#                                         random_state=RANDOM_STATE)
# gbdt_model.fit(X_train_smote, y_train_smote)

# gbdt_pred = gbdt_model.predict(X_test_scaled)
# print("GBDT Result")
# print(classification_report(y_test, gbdt_pred, zero_division=0))
# SMOTE better

# --------------------------------------------ET-------------------------------------------
# from sklearn.ensemble import ExtraTreesClassifier

# et_model = ExtraTreesClassifier(n_estimators=100,    # Number of trees in the ensemble
#                                 max_depth=5,         # Maximum depth of each individual tree
#                                 random_state=RANDOM_STATE,
#                                 n_jobs=-1, class_weight='balanced')           # Use all cores for faster computation
# et_model.fit(X_train_scaled, y_train)

# et_pred = et_model.predict(X_test_scaled)
# print("ET Result")
# print(classification_report(y_test, et_pred, zero_division=0))
# SMOTE worse

# --------------------------------------SVC---------------------------------------
# from sklearn.svm import SVC

# svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE, class_weight='balanced')
# svc_model.fit(X_train_smote, y_train_smote)  # Ensure target is binary

# svc_pred = svc_model.predict(X_test_scaled)
# print("SVC Result")
# print(classification_report(y_test, svc_pred, zero_division=0))
# No difff








































# # With SMOTE now

# RF_model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
# RF_model.fit(X_train_smote, y_train_smote)

# RF_pred = RF_model.predict(X_test)
# print("RF SMOTE Result")
# print(classification_report(y_test, RF_pred, zero_division=0))

















# ----------------RESULTS-----------------
# All three models give an accuracy of 0.78 and they only seem to work for non violent crime
# This is because there is a huge imbalance in the crime data so i will now need to try and use SMOTE or smthn else to balance
# Scaled data and not scaled data makes no difference except for on LR where it makes it converge before max iterations
# But it still ends up with the same accuracy and predictions
# -------------------------------------------------







# ------------------------------------------LR----------------------------------------------------

from sklearn.linear_model import LogisticRegression

# # Train the model
# LR_model = LogisticRegression(max_iter = 500, random_state = RANDOM_STATE)
# LR_model.fit(X_train_scaled, y_train)

# # Evaluate the model
# LR_pred = LR_model.predict(X_test_scaled)
# print(classification_report(y_test, LR_pred))
# VEry bad



# ---------------------------LR SMOTE BALANCED----------------------------------

# # Train the model
# LR_model = LogisticRegression(max_iter = 500, random_state = RANDOM_STATE)
# LR_model.fit(X_train_smote, y_train_smote)

# # Evaluate the model
# LR_pred = LR_model.predict(X_test_scaled)
# print(classification_report(y_test, LR_pred))














# ----------------------------------MLP-----------------------------------

from sklearn.neural_network import MLPClassifier

# # Train the model
# MLP_model = MLPClassifier(hidden_layer_sizes = (50,), solver='adam', learning_rate_init = 0.001, max_iter = 200, random_state = RANDOM_STATE)
# MLP_model.fit(X_train_scaled, y_train)
# print("Finished training MLP\n")

# # Evaluate the model
# MLP_pred = MLP_model.predict(X_test_scaled)
# print("Finished evaluating MLP\n")
# print(classification_report(y_test, MLP_pred))


# ----------------------------RF smote-------------------------

# Train the model
# RF_model = RandomForestClassifier(class_weight='balanced',random_state=RANDOM_STATE)
# RF_model.fit(X_train_smote, y_train_smote)
# print("Finished training RF\n")

# # Evaluate the model
# RF_pred = RF_model.predict(X_test_scaled)
# print("Finished evaluating RF\n")
# print(classification_report(y_test, RF_pred))






# # Step 6: Train a Decision Tree Classifier
# clf = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5, min_samples_split=10)
# clf.fit(X_train, y_train)

# # Step 7: Evaluate the model
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))


# print(y_train.value_counts())
# print(y_test.value_counts())

# print(confusion_matrix(y_test, y_pred))

# # Step 6: Train a Random Forest Classifier
# rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)  # n_estimators is the number of trees
# rf_clf.fit(X_train, y_train)

# # Step 7: Evaluate the model
# y_pred = rf_clf.predict(X_test)
# print(classification_report(y_test, y_pred))


# # Train the model
# lr_clf = LogisticRegression(max_iter=1000, random_state=42)
# lr_clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = lr_clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Train the model
# knn_clf = KNeighborsClassifier(n_neighbors=5)
# knn_clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = knn_clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Train the model
# svm_clf = SVC(kernel='linear', random_state=42)
# svm_clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = svm_clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Train the model
# gb_clf = GradientBoostingClassifier(random_state=42)
# gb_clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = gb_clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # # Train the model
# # xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# # xgb_clf.fit(X_train, y_train)

# # # Evaluate the model
# # y_pred = xgb_clf.predict(X_test)
# # print(classification_report(y_test, y_pred))

# # Train the model
# nb_clf = MultinomialNB()
# nb_clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = nb_clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Train the model
# nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
# nn_clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = nn_clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Train the model
# ada_clf = AdaBoostClassifier(random_state=42)
# ada_clf.fit(X_train, y_train)

# # Evaluate the model
# y_pred = ada_clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Train the model (example: clustering neighborhoods based on crime data)
# kmeans = KMeans(n_clusters=5, random_state=42)
# kmeans.fit(X)

# # Show clusters
# print(kmeans.labels_)


# Corr analysis
# correlation_matrix = CensusData.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.show()


# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# feature_importances = model.feature_importances_
# feature_names = X_train.columns
# feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# print(feature_importance_df)



# # Cor
# data = pd.merge(CensusData, CrimeData, on='Name')
# # Drop non-numeric columns or convert them
# data_numeric = data.select_dtypes(include=[float, int])
# correlation_matrix = data_numeric.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
# plt.show()