import pandas as pd
import matplotlib.pyplot as plt

data_path = '/home/campus.ncl.ac.uk/b3024896/Projects/gym-donkeytrack/logs/donkey_300k/target_data.csv'

# read in csv data
df = pd.read_csv(data_path, header=None)

# look at first few rows
print(df.head(5))

# check for any NaNs in dataframe - REMEMBER TO ADD 1 TO GET ROW NUMBER AND IMAGE FILENAME
print(pd.isnull(df).any(1).nonzero()[0])

# pull out columns
r_col = df.iloc[:, 0]
psi_col = df.iloc[:, 1]
phi_col = df.iloc[:, 2]

# get minimum and maximum values
max_r = r_col.max()
min_r = r_col.min()

max_psi = psi_col.max()
min_psi = psi_col.min()

max_phi = phi_col.max()
min_phi = phi_col.min()

print('R: min = {}, max = {}, Psi: min = {}, max = {}, Phi: min = {}, max = {}'.format(min_r, max_r, min_psi, max_psi, min_phi, max_phi))

# see spread of values with histogram
r_col.plot.hist()
plt.show()

psi_col.plot.hist()
plt.show()

phi_col.plot.hist()
plt.show()

# once decided on ranges want to use based on min, max and spread, check which rows are outside of range (edit as needed) 
print('Index of rows with R > 30: {}'.format(df[df.iloc[:, 0] > 30].index + 1))

print('Index of rows with Psi < -90: {}'.format(df[df.iloc[:, 1] < -90].index + 1))

print('Index of rows with Psi > 90: {}'.format(df[df.iloc[:, 1] > 90].index + 1))

print('Index of rows with Phi < 0: {}'.format(df[df.iloc[:, 2] < 0].index + 1))

print('Index of rows with Phi > 90: {}'.format(df[df.iloc[:, 2] > 90].index + 1))






