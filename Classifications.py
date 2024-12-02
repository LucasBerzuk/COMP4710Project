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
    'No Certification Ratio',
    'Employed Ratio', 
    'Pop. Density (per km^2)', 
    'Avg. Income', 
    'Ownership Ratio', 
    'High School Diploma Ratio', 
    'Indigenous Ratio', 
    'Immigrant Ratio', 
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer

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

# Dictionary to store results
model_results = {}

# Define a function to train, predict, and store results
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    model_results[model_name] = report

# -----------------------------------------RF------------------------------------ Optimal solution so far
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 1000,
                                  criterion= 'entropy',
                                  max_depth = 5,
                                  min_samples_split = 2,
                                  min_samples_leaf = 4,
                                  min_weight_fraction_leaf = 0,
                                  max_features = 'sqrt',
                                  max_leaf_nodes = None,# 7 is more balanced
                                  min_impurity_decrease = 0,
                                  bootstrap = True,
                                  oob_score = False,
                                  random_state=RANDOM_STATE, 
                                  class_weight='balanced',
                                  n_jobs = None,
                                  verbose = 0,
                                  warm_start = False,
                                  ccp_alpha = 0,
                                  max_samples = None)
evaluate_model(RF_model, "Random Forest", X_train, y_train, X_test, y_test)
# RF_model.fit(X_train, y_train)

# RF_pred = RF_model.predict(X_test)
# print("RF Result")
# print(classification_report(y_test, RF_pred, zero_division=0))

# --------------------------------------MLP-NN-------------------------------------Optimal so far
from sklearn.neural_network import MLPClassifier

MLP_model = MLPClassifier(hidden_layer_sizes=(100,100),  # One hidden layer with 100 neurons
                          solver='adam',             # Optimizer (Adam)
                          activation='relu',         # ReLU activation function
                          max_iter=1500,             # Number of iterations for training
                          random_state=RANDOM_STATE, 
                          alpha = 0.0383,
                          batch_size = "auto",
                          learning_rate = "constant",
                          learning_rate_init = 0.001,
                          power_t = 0.5,
                          shuffle = True,
                          tol = 0.0001,
                          verbose = False,
                          warm_start = False,
                          momentum = 0.9,
                          nesterovs_momentum = True,
                          early_stopping = False,
                          validation_fraction = 0.1,
                          beta_1 = 0.9,
                          beta_2 = 0.999,
                          epsilon = 0.00000001,
                          n_iter_no_change = 10,
                          max_fun = 15000)
evaluate_model(MLP_model, "MLP Neural Network", X_train_smote, y_train_smote, X_test, y_test)

# MLP_model.fit(X_train_smote, y_train_smote)

# MLP_pred = MLP_model.predict(X_test)
# print("MLP Result")
# print(classification_report(y_test, MLP_pred, zero_division=0))

# ---------------------------------------K-NN-----------------------------------------Optimal so far
from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors=3,  # Number of neighbors (k)
                                 metric='minkowski',  # Distance metric (Minkowski is the default)
                                 p=3,                 # p=2 corresponds to Euclidean distance
                                 n_jobs=None,
                                 weights = "uniform",
                                 algorithm = "auto",
                                 leaf_size = 30,
                                 metric_params = None)
evaluate_model(KNN_model, "K-Nearest Neighbors", X_train_scaled, y_train, X_test_scaled, y_test)

# KNN_model.fit(X_train_scaled, y_train)

# KNN_pred = KNN_model.predict(X_test_scaled)
# print("KNN Result")
# print(classification_report(y_test, KNN_pred, zero_division=0))

# --------------------------------------DT------------------------------------------ current optimal
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE,
                                  criterion='gini',   # Use Gini impurity for decision splits (default)
                                  max_depth=10,        # Limit the depth of the tree to prevent overfitting
                                  min_samples_split=10, # Minimum samples required to split a node
                                  min_samples_leaf=5,
                                  class_weight='balanced',
                                  splitter = "best",
                                  min_weight_fraction_leaf = 0,
                                  max_features = None,
                                  max_leaf_nodes = None,
                                  min_impurity_decrease = 0.001,
                                  ccp_alpha = 0.005)
evaluate_model(dt_model, "Decision Tree", X_train_smote, y_train_smote, X_test, y_test)

# dt_model.fit(X_train_smote, y_train_smote)

# dt_pred = dt_model.predict(X_test)
# print("DT Result")
# print(classification_report(y_test, dt_pred, zero_division=0))

# ------------------------------------------GBDT---------------------------------------- Current Optimal
from sklearn.ensemble import GradientBoostingClassifier

gbdt_model = GradientBoostingClassifier(n_estimators=9,    # Number of trees in the ensemble
                                        learning_rate=0.1,   # Learning rate to shrink contribution of each tree
                                        max_depth=5,         # Maximum depth of each individual tree
                                        random_state=RANDOM_STATE,
                                        loss = "log_loss",
                                        subsample = 1,
                                        criterion = "squared_error",
                                        min_samples_split = 3,
                                        min_samples_leaf = 1,
                                        min_weight_fraction_leaf = 0,
                                        min_impurity_decrease = 0,
                                        init = None,
                                        max_features = None,
                                        verbose = 0,
                                        max_leaf_nodes = None,
                                        warm_start = False,
                                        validation_fraction = 0.1,
                                        n_iter_no_change = None,
                                        tol = 0.0001,
                                        ccp_alpha = 0.001)
evaluate_model(gbdt_model, "Gradient Boosting", X_train_smote, y_train_smote, X_test, y_test)

# gbdt_model.fit(X_train_smote, y_train_smote)

# gbdt_pred = gbdt_model.predict(X_test)
# print("GBDT Result")
# print(classification_report(y_test, gbdt_pred, zero_division=0))

# --------------------------------------------ET------------------------------------------- Current Optimal
from sklearn.ensemble import ExtraTreesClassifier

et_model = ExtraTreesClassifier(n_estimators=100,    # Number of trees in the ensemble
                                max_depth=20,         # Maximum depth of each individual tree
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                                class_weight='balanced',
                                criterion = 'gini',
                                min_samples_split = 2,
                                min_samples_leaf = 1,
                                min_weight_fraction_leaf = 0,
                                max_features = 'sqrt',
                                max_leaf_nodes = None,
                                min_impurity_decrease= 0,
                                bootstrap = True,
                                oob_score = False,
                                verbose = 0,
                                warm_start = False,
                                ccp_alpha = 0,
                                max_samples = None)     
evaluate_model(et_model, "Extra Trees", X_train_scaled, y_train, X_test_scaled, y_test)
   
# et_model.fit(X_train, y_train)

# et_pred = et_model.predict(X_test)
# print("ET Result")
# print(classification_report(y_test, et_pred, zero_division=0))

# --------------------------------------SVC--------------------------------------- Optimal currently
from sklearn.svm import SVC

svc_model = SVC(kernel='linear', 
                C=0.5, 
                gamma='scale', 
                random_state=RANDOM_STATE, 
                class_weight='balanced',
                degree = 3,
                coef0 = 0,
                shrinking = True,
                probability = False,
                tol = 0.001,
                cache_size = 200,
                verbose = False,
                max_iter = 1000,
                decision_function_shape = "ovr",
                break_ties = True)
evaluate_model(svc_model, "Support Vector Classifier", X_train_scaled, y_train, X_test_scaled, y_test)

# svc_model.fit(X_train_scaled, y_train)  # Ensure target is binary

# svc_pred = svc_model.predict(X_test_scaled)
# print("SVC Result")
# print(classification_report(y_test, svc_pred, zero_division=0))

# ------------------------------------ Results ----------------------------------
def process_report(report):
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    # Round numeric values to 3 decimal places
    return df.round(3)

# Store formatted DataFrames in a dictionary
formatted_results = {model: process_report(report) for model, report in model_results.items()}

# Print results for all models
for model, df in formatted_results.items():
    print(f"Results for {model}:")
    print(df)
    print("\n")

# -----------------------EXPORTING TABLES IF NEEDED---------------------
# import matplotlib.pyplot as plt

# # Assuming `formatted_results` is created as in the previous example
# for model, df in formatted_results.items():
#     fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the figure size as needed
#     ax.axis('tight')  # Turn off axes
#     ax.axis('off')  # Turn off axes

#     # Add a table at the axes
#     table = ax.table(cellText=df.reset_index().values,  # Data for the table
#                      colLabels=["Class"] + df.columns.tolist(),  # Column headers
#                      loc='center',  # Position table at the center
#                      cellLoc='center')  # Align text to center
    
#     # Format the table
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.auto_set_column_width(col=list(range(len(df.columns) + 1)))
    
#     # Set title
#     ax.set_title(f"Classification Report: {model}", fontsize=14, pad=20)

#     # Show the plot
#     plt.show()





#  ---------------------------------
# PLOTTING
# ----------------------------------
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.decomposition import PCA

# # Plot feature importance
# def plot_feature_importance(model, feature_names):
#     importance = model.feature_importances_
#     sorted_idx = importance.argsort()
#     plt.figure(figsize=(10, 6))
#     plt.barh(range(len(sorted_idx)), importance[sorted_idx], align="center")
#     plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
#     plt.title("Feature Importance")
#     plt.show()

# plot_feature_importance(RF_model, features)

# # Confusion Matrix
# def plot_confusion_matrix(y_true, y_pred, labels):
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#     disp.plot(cmap="Blues", values_format="d")
#     plt.title("Confusion Matrix")
#     plt.show()

# plot_confusion_matrix(y_test, RF_pred, labels=RF_model.classes_)

# # Visualizing Clusters with PCA
# def plot_pca(X, y, title="PCA of Socioeconomic Features"):
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(
#         x=X_pca[:, 0],
#         y=X_pca[:, 1],
#         hue=y,
#         palette="viridis",
#         s=50,
#         alpha=0.8
#     )
#     plt.title(title)
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.legend(title="Crime Rate")
#     plt.grid()
#     plt.show()

# plot_pca(X_test_scaled, y_test, title="PCA of Testing Data (Post-Scaling)")

# # Precision, Recall, and F1 Scores per Class
# def plot_classification_report(report):
#     report_data = []
#     for label, metrics in report.items():
#         if isinstance(metrics, dict):
#             report_data.append({"class": label, **metrics})
#     df_report = pd.DataFrame(report_data)
#     df_report.set_index("class", inplace=True)
#     df_report[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(10, 6), cmap="viridis")
#     plt.title("Classification Report Metrics")
#     plt.ylabel("Score")
#     plt.xticks(rotation=45)
#     plt.grid(axis="y")
#     plt.show()

# # Convert classification report to dict and plot
# report_dict = classification_report(y_test, RF_pred, output_dict=True, zero_division=0)
# plot_classification_report(report_dict)