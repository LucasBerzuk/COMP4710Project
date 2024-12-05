import pandas as pd
import joblib

# create pickel files to load data quickly
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

newData = MoreCrime.groupby(["Month of Report Date", "Name"]).sum().reset_index()

months13 = ["Jan-13", "Feb-13", "Mar-13", "Apr-13", "May-13", "Jun-13", "Jul-13", "Aug-13", "Sep-13", "Oct-13", "Nov-13", "Dec-13"]
data13 = newData[newData['Month of Report Date'].isin(months13)]
data13Grouped = data13.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data13Grouped)

months14 = ["Jan-14", "Feb-14", "Mar-14", "Apr-14", "May-14", "Jun-14", "Jul-14", "Aug-14", "Sep-14", "Oct-14", "Nov-14", "Dec-14"]
data14 = newData[newData['Month of Report Date'].isin(months14)]
data14Grouped = data14.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data14Grouped)

months15 = ["Jan-15", "Feb-15", "Mar-15", "Apr-15", "May-15", "Jun-15", "Jul-15", "Aug-15", "Sep-15", "Oct-15", "Nov-15", "Dec-15"]
data15 = newData[newData['Month of Report Date'].isin(months15)]
data15Grouped = data15.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data15Grouped)

months16 = ["Jan-16", "Feb-16", "Mar-16", "Apr-16", "May-16", "Jun-16", "Jul-16", "Aug-16", "Sep-16", "Oct-16", "Nov-16", "Dec-16"]
data16 = newData[newData['Month of Report Date'].isin(months16)]
data16Grouped = data16.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data16Grouped)

months17 = ["Jan-17", "Feb-17", "Mar-17", "Apr-17", "May-17", "Jun-17", "Jul-17", "Aug-17", "Sep-17", "Oct-17", "Nov-17", "Dec-17"]
data17 = newData[newData['Month of Report Date'].isin(months17)]
data17Grouped = data17.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data17Grouped)

months18 = ["Jan-18", "Feb-18", "Mar-18", "Apr-18", "May-18", "Jun-18", "Jul-18", "Aug-18", "Sep-18", "Oct-18", "Nov-18", "Dec-18"]
data18 = newData[newData['Month of Report Date'].isin(months18)]
data18Grouped = data18.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data18Grouped)

months19 = ["Jan-19", "Feb-19", "Mar-19", "Apr-19", "May-19", "Jun-19", "Jul-19", "Aug-19", "Sep-19", "Oct-19", "Nov-19", "Dec-19"]
data19 = newData[newData['Month of Report Date'].isin(months19)]
data19Grouped = data19.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data19Grouped)

months20 = ["Jan-20", "Feb-20", "Mar-20", "Apr-20", "May-20", "Jun-20", "Jul-20", "Aug-20", "Sep-20", "Oct-20", "Nov-20", "Dec-20"]
data20 = newData[newData['Month of Report Date'].isin(months20)]
data20Grouped = data20.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data20Grouped)

months21 = ["Jan-21", "Feb-21", "Mar-21", "Apr-21", "May-21", "Jun-21", "Jul-21", "Aug-21", "Sep-21", "Oct-21", "Nov-21", "Dec-21"]
data21 = newData[newData['Month of Report Date'].isin(months21)]
data21Grouped = data21.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data21Grouped)

months22 = ["Jan-22", "Feb-22", "Mar-22", "Apr-22", "May-22", "Jun-22", "Jul-22", "Aug-22", "Sep-22", "Oct-22", "Nov-22", "Dec-22"]
data22 = newData[newData['Month of Report Date'].isin(months22)]
data22Grouped = data22.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data22Grouped)

months23 = ["Jan-23", "Feb-23", "Mar-23", "Apr-23", "May-23", "Jun-23", "Jul-23", "Aug-23", "Sep-23", "Oct-23", "Nov-23", "Dec-23"]
data23 = newData[newData['Month of Report Date'].isin(months23)]
data23Grouped = data23.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data23Grouped)

months24 = ["Jan-24", "Feb-24", "Mar-24", "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24", "Sep-24", "Oct-24", "Nov-24", "Dec-24"]
data24 = newData[newData['Month of Report Date'].isin(months24)]
data24Grouped = data24.groupby("Name").agg({"Count": "sum"}).reset_index()
pd.set_option("display.max_rows", None)
# print(data24Grouped)


merged23 = pd.merge(data23Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
merged22 = pd.merge(data22Grouped[['Name', 'Count']], CensusData, on="Name", how="right")
combined_df = pd.concat([merged23, merged22], axis=0, ignore_index=True)
print(combined_df)