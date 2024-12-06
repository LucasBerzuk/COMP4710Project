import pandas as pd
import joblib

# --------------------------------------------------------
# Create Pickle file to load data faster
# --------------------------------------------------------
try:
  CensusData = joblib.load("CensusNeighbourhoods.pkl")
  CrimeData = joblib.load("CrimeByNeighbourhood.pkl")
  MoreCrime = joblib.load("MoreCrime.pkl")
  print("Loaded from cache\n")

except FileNotFoundError:
  CensusData = pd.read_csv("CensusNeighbourhoods.csv")
  CrimeData = pd.read_csv("CrimeByNeighbourhood.csv")
  MoreCrime = pd.read_csv("MoreCrime.csv")
  joblib.dump(CensusData, "CensusNeighbourhoods.pkl")
  joblib.dump(CrimeData, "CrimeByNeighbourhood.pkl")
  joblib.dump(MoreCrime, "MoreCrime.pkl")
  print("Loaded from CSV and cached\n")
# -----------------------------------------------------------


# -----------------------------------------------------------
# Editing the census data
# -----------------------------------------------------------

# All original features
all_indicators = [
  'Pop.', 'Pop. Density (per km^2)', 'Male Pop.', 'Female Pop.', 'Indigenous Identity',
  'Visible Minorities', 'Immigrants (place of birth)', 'No Certificate, Diploma or Degree',
  'High School Diploma or Equiv.', 'Post Secondary Certificate, Diploma or Degree',
  'Employed', 'Unemployed', 'Avg. Income', 'LICO-AT', 'Avg. Person per House', 
  'Avg. Household Income', 'Dwelling-Owned', 'Dwelling-Rented'
]

# Ensure all relevant columns are numeric
CensusData[all_indicators] = CensusData[all_indicators].apply(pd.to_numeric, errors='coerce')

# Function to compute ratios with a check for zero in the denominator or numerator
def safe_ratio(numerator, denominator):
    return 0 if denominator == 0 or numerator == 0 else numerator / denominator

# Normalize the features because they are dependent on population and other variables
# Using the safe ratio to not have any errors
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

# List of the features we will start with for our model
all_computed_indicators = [
  'High School Diploma Ratio', 'Post Secondary Ratio', 'Employed Ratio',
  'Avg. Income', 'Household Income Ratio', 'Ownership Ratio', 'Pop. Density (per km^2)', 
  'Indigenous Ratio', 'Minority Ratio', 'Immigrant Ratio', 
  'No Certification Ratio', 'LICO Ratio', 'Male Ratio', 'Female Ratio'
]

# Make a new DF with these features
data_computed = pd.DataFrame({'Name': CensusData['Name']})
data_computed[all_computed_indicators] = CensusData[all_computed_indicators]
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# Breaking down MoreData (more crime data we imported) so that
# We can have multiple instances of neighbourhoods in our dataset
# With different crime rates to allow for more data to test on
# -----------------------------------------------------------------------------------

newData = MoreCrime.groupby(["Month of Report Date", "Name"]).sum().reset_index()

# Putting each year of data into an individual dataset
months13 = ["Jan-13", "Feb-13", "Mar-13", "Apr-13", "May-13", "Jun-13", "Jul-13", "Aug-13", "Sep-13", "Oct-13", "Nov-13", "Dec-13"]
data13 = newData[newData['Month of Report Date'].isin(months13)]
data13Grouped = data13.groupby("Name").agg({"Count": "sum"}).reset_index()

months14 = ["Jan-14", "Feb-14", "Mar-14", "Apr-14", "May-14", "Jun-14", "Jul-14", "Aug-14", "Sep-14", "Oct-14", "Nov-14", "Dec-14"]
data14 = newData[newData['Month of Report Date'].isin(months14)]
data14Grouped = data14.groupby("Name").agg({"Count": "sum"}).reset_index()

months15 = ["Jan-15", "Feb-15", "Mar-15", "Apr-15", "May-15", "Jun-15", "Jul-15", "Aug-15", "Sep-15", "Oct-15", "Nov-15", "Dec-15"]
data15 = newData[newData['Month of Report Date'].isin(months15)]
data15Grouped = data15.groupby("Name").agg({"Count": "sum"}).reset_index()

months16 = ["Jan-16", "Feb-16", "Mar-16", "Apr-16", "May-16", "Jun-16", "Jul-16", "Aug-16", "Sep-16", "Oct-16", "Nov-16", "Dec-16"]
data16 = newData[newData['Month of Report Date'].isin(months16)]
data16Grouped = data16.groupby("Name").agg({"Count": "sum"}).reset_index()

months17 = ["Jan-17", "Feb-17", "Mar-17", "Apr-17", "May-17", "Jun-17", "Jul-17", "Aug-17", "Sep-17", "Oct-17", "Nov-17", "Dec-17"]
data17 = newData[newData['Month of Report Date'].isin(months17)]
data17Grouped = data17.groupby("Name").agg({"Count": "sum"}).reset_index()

months18 = ["Jan-18", "Feb-18", "Mar-18", "Apr-18", "May-18", "Jun-18", "Jul-18", "Aug-18", "Sep-18", "Oct-18", "Nov-18", "Dec-18"]
data18 = newData[newData['Month of Report Date'].isin(months18)]
data18Grouped = data18.groupby("Name").agg({"Count": "sum"}).reset_index()

months19 = ["Jan-19", "Feb-19", "Mar-19", "Apr-19", "May-19", "Jun-19", "Jul-19", "Aug-19", "Sep-19", "Oct-19", "Nov-19", "Dec-19"]
data19 = newData[newData['Month of Report Date'].isin(months19)]
data19Grouped = data19.groupby("Name").agg({"Count": "sum"}).reset_index()

months20 = ["Jan-20", "Feb-20", "Mar-20", "Apr-20", "May-20", "Jun-20", "Jul-20", "Aug-20", "Sep-20", "Oct-20", "Nov-20", "Dec-20"]
data20 = newData[newData['Month of Report Date'].isin(months20)]
data20Grouped = data20.groupby("Name").agg({"Count": "sum"}).reset_index()

months21 = ["Jan-21", "Feb-21", "Mar-21", "Apr-21", "May-21", "Jun-21", "Jul-21", "Aug-21", "Sep-21", "Oct-21", "Nov-21", "Dec-21"]
data21 = newData[newData['Month of Report Date'].isin(months21)]
data21Grouped = data21.groupby("Name").agg({"Count": "sum"}).reset_index()

months22 = ["Jan-22", "Feb-22", "Mar-22", "Apr-22", "May-22", "Jun-22", "Jul-22", "Aug-22", "Sep-22", "Oct-22", "Nov-22", "Dec-22"]
data22 = newData[newData['Month of Report Date'].isin(months22)]
data22Grouped = data22.groupby("Name").agg({"Count": "sum"}).reset_index()

months23 = ["Jan-23", "Feb-23", "Mar-23", "Apr-23", "May-23", "Jun-23", "Jul-23", "Aug-23", "Sep-23", "Oct-23", "Nov-23", "Dec-23"]
data23 = newData[newData['Month of Report Date'].isin(months23)]
data23Grouped = data23.groupby("Name").agg({"Count": "sum"}).reset_index()

months24 = ["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24", "Oct-24", "Nov-24", "Dec-24"]
data24 = newData[newData['Month of Report Date'].isin(months24)]
data24Grouped = data24.groupby("Name").agg({"Count": "sum"}).reset_index()

# Merge the data with the census data and drop and NaN for neighbourhoods that 
# either did not exist yet or just have no data for that specific year
merged13 = pd.merge(data13Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged13 = merged13.dropna(subset=['Count'])
merged14 = pd.merge(data14Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged14 = merged14.dropna(subset=['Count'])
merged15 = pd.merge(data15Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged15 = merged15.dropna(subset=['Count'])
merged16 = pd.merge(data16Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged16 = merged16.dropna(subset=['Count'])
merged17 = pd.merge(data17Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged17 = merged17.dropna(subset=['Count'])
merged18 = pd.merge(data18Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged18 = merged18.dropna(subset=['Count'])
merged19 = pd.merge(data19Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged19 = merged19.dropna(subset=['Count'])
merged20 = pd.merge(data20Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged20 = merged20.dropna(subset=['Count'])
merged21 = pd.merge(data21Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged21 = merged21.dropna(subset=['Count'])
merged22 = pd.merge(data22Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged22 = merged22.dropna(subset=['Count'])
merged23 = pd.merge(data23Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged23 = merged23.dropna(subset=['Count'])
merged24 = pd.merge(data24Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged24 = merged24.dropna(subset=['Count'])

# Option set for output
pd.set_option("display.max_rows", None)

# Put the data into one dataset so that we can have more than 196 rows to test on like we origianlly had with 'CrimeData'
finalData = pd.concat([merged13, merged14, merged15, merged16, merged17, merged18, merged19, merged20, merged21, merged22, merged23], axis=0, ignore_index=True)

# Need to trim heavy outliers, there are alot of these that are effecting results
# Calculate Q1 (25th percentile) and Q3 (75th percentile) for each group
q1 = finalData.groupby('Name')['Count'].quantile(0.25)
q3 = finalData.groupby('Name')['Count'].quantile(0.75)

# Calculate IQR
iqr = q3 - q1

# Define outlier thresholds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Merge thresholds with original data
outlier_bounds = pd.DataFrame({'Lower': lower_bound, 'Upper': upper_bound}).reset_index()
finalData_with_bounds = finalData.merge(outlier_bounds, on='Name')

# Filter out rows with 'Count' outside the bounds
trimmed_data = finalData_with_bounds[
    (finalData_with_bounds['Count'] >= finalData_with_bounds['Lower']) &
    (finalData_with_bounds['Count'] <= finalData_with_bounds['Upper'])
]

# Drop intermediate columns, now we have our dataset to run models on
trimmed_data = trimmed_data.drop(columns=['Lower', 'Upper'])
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Classifications
# -----------------------------------------------------------------------------

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

RANDOM_STATE = 942


# This is the bounds for what is low and high, 1 crime a week is what we consider high
# High is 52+ because each instance of a neighbourhood in the dataset is based on a 1 year time frame
# So some neighbourhoods will have up to 11 instances in the data
def categorize_crime_rate(count):
    if count <= 51:
        return 'Low'
    else:
        return 'High'

# Apply the function to create the 'Crime Rate' column
trimmed_data['Crime Rate'] = trimmed_data['Count'].apply(categorize_crime_rate)

# Set features, these are the 6 found that were the least correlated, if we want to add features back
# they can be grabbed from 'all_computed_indicators' on line 87
features = [
    # 'No Certification Ratio',
    # 'Employed Ratio', 
    # 'Pop. Density (per km^2)', 
    # 'Avg. Income', 
    # 'Ownership Ratio', 
    # 'Minority Ratio',
    'High School Diploma Ratio', 'Post Secondary Ratio', 'Employed Ratio',
  'Avg. Income', 'Household Income Ratio', 'Ownership Ratio', 'Pop. Density (per km^2)', 
  'Indigenous Ratio', 'Minority Ratio', 'Immigrant Ratio', 
  'No Certification Ratio', 'LICO Ratio', 'Male Ratio', 'Female Ratio'
]

# Now set the target variable to crime rate
target = 'Crime Rate'

# Split the data into features and target
X = trimmed_data[features] 
y = trimmed_data[target]  

# Split data into training and testing sets, 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = RANDOM_STATE)

# Ensure that both X_train and X_test are DataFrames to retain feature names
X_train = pd.DataFrame(X_train, columns=features)
X_test = pd.DataFrame(X_test, columns=features)

# Methods of balancing-------------------------------------------
# ------------------------Scaling--------------------------------
# Instantiate the scaler
scaler = RobustScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------SMOTE------------------------------
# Initialize SMOTE
smote = SMOTE(random_state = RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)  # Resample training data

# Ensure the resampled data is still in DataFrame format to avoid issues with model predictions
X_train_smote = pd.DataFrame(X_train_smote, columns=features)
# ---------------------------------------------------------------

# Output Formatting----------------------------------------------
# Dictionary to store results
model_results = {}

# Define a function to train, predict, and store results
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    model_results[model_name] = report
# -----------------------------------------------------------------


# -------------------------------MLP-NN----------------------------
MLP_model = MLPClassifier(random_state = RANDOM_STATE,
                          hidden_layer_sizes = (1000,500))
evaluate_model(MLP_model, "MLP Neural Network", X_train_scaled, y_train, X_test_scaled, y_test)
MLP_pred = MLP_model.predict(X_test_scaled) # Generate this here for the plots

# -------------------------------K-NN------------------------------
KNN_model = KNeighborsClassifier(n_neighbors = 5,
                                 weights = "distance",
                                 algorithm = "brute")
evaluate_model(KNN_model, "K-Nearest Neighbors", X_train, y_train, X_test, y_test)
KNN_pred = KNN_model.predict(X_test) # Generate this here for the plots

# --------------------------------DT--------------------------------
DT_model = DecisionTreeClassifier(random_state = RANDOM_STATE)
evaluate_model(DT_model, "Decision Tree", X_train, y_train, X_test, y_test)
DT_pred = DT_model.predict(X_test) # Generate this here for the plots

# --------------------------------GBDT------------------------------
GBDT_model = GradientBoostingClassifier(random_state = RANDOM_STATE)
evaluate_model(GBDT_model, "Gradient Boosting", X_train, y_train, X_test, y_test)
GBDT_pred = GBDT_model.predict(X_test) # Generate this here for the plots

# ---------------------------------RF------------------------------
RF_model = RandomForestClassifier(random_state = RANDOM_STATE)
evaluate_model(RF_model, "Random Forest", X_train, y_train, X_test, y_test)
RF_pred = RF_model.predict(X_test) # Generate this here for the plots

# ---------------------------------ET-------------------------------
ET_model = ExtraTreesClassifier(random_state = RANDOM_STATE)         
evaluate_model(ET_model, "Extra Trees", X_train, y_train, X_test, y_test)
ET_pred = ET_model.predict(X_test) # Generate this here for the plots

# ---------------------------------SVC------------------------------
SVC_model = SVC(random_state = RANDOM_STATE,
                C = 1000)
evaluate_model(SVC_model, "Support Vector Classifier", X_train_scaled, y_train, X_test_scaled, y_test)
SVC_pred = SVC_model.predict(X_test_scaled) # Generate this here for the plots

# ------------------------------- Results ---------------------------
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
# ----------------------------------------------------------------------



# # -----------------------EXPORTING TABLES IF NEEDED---------------------
# import matplotlib.pyplot as plt

# # Assuming `formatted_results` is a dictionary of DataFrames with classification reports
# for model, df in formatted_results.items():
#     # Drop the 'support' column
#     if "support" in df.columns:
#         df = df.drop(columns=["support"])
    
#     # Drop 'weighted avg' and 'macro avg' rows if they exist
#     df = df.drop(index=["weighted avg", "macro avg"], errors="ignore")
    
#     # Plotting
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






# -----------------------------------------------
# PLOTTING
# -----------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# Plot feature importance
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title("Feature Importance")
    plt.show()

# plot_feature_importance(MLP_model, features)
# plot_feature_importance(KNN_model, features)
plot_feature_importance(DT_model, features)
plot_feature_importance(GBDT_model, features)
# plot_feature_importance(RF_model, features)
plot_feature_importance(ET_model, features)
# plot_feature_importance(SVC_model, features)

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