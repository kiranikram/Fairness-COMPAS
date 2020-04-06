import torch.nn as nn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from time import time
from torch import optim

"""# Dataset Creation

## Dataframe from merged csv files 
df_final: without feature engineering of priors

df_final_priors: with feature engineering of priors
"""

path = "./"
filename_read = os.path.join(path,"compas-scores-two-years_original.csv")

df_two_years = pd.read_csv(filename_read, usecols = ["first",
                                        "last",
                                        "sex",
                                        "dob",
                                        "race",
                                        "juv_fel_count",
                                        "juv_misd_count",
                                        "juv_other_count",
                                        "priors_count",
                                        "days_b_screening_arrest",
                                        "compas_screening_date",
                                        "c_case_number",
                                        "c_offense_date",
                                        "c_charge_degree",
                                        "is_recid",
                                        "is_violent_recid",
                                        "r_charge_degree",
                                        "vr_charge_degree",
                                        ])


#to include marital status of defendents, importing raw data for all set of defendents
filename_read = os.path.join(path,"compas-scores-raw_original.csv")
df_raw = pd.read_csv(filename_read, usecols = ["FirstName",
                                        "LastName",
                                        "DateOfBirth",
                                        "MaritalStatus"
                                        ])

#preprocessing step, to ensure no missing values
df_two_years = df_two_years.dropna(subset=["compas_screening_date", "first","last", "dob"])
df_raw = df_raw.dropna(subset=["FirstName","LastName","DateOfBirth"])

#when reduced to the four features we use, the raw file has many duplicate rows, so we drop them
df_raw = df_raw.drop_duplicates()

#converting type object to type date
df_two_years["compas_screening_date"] = pd.to_datetime(df_two_years["compas_screening_date"])
df_two_years["dob"] = pd.to_datetime(df_two_years["dob"])

#changing column names as to easily merge two datasets
df_raw = df_raw.rename(columns={"LastName" : "last", "FirstName" : "first", "DateOfBirth" : "dob"})
df_raw["dob"] = pd.to_datetime(df_raw["dob"])
#ensuring all elements in the column are in lower case, so that when merging, they match the two years versions
df_raw["last"] = df_raw["last"].str.lower()
df_raw["first"] = df_raw["first"].str.lower()

#to ensure dates were all in the same format, anything over the date the dataset was tested for needed to be
#subtracted by 100 years.
#certain values in the dob column were displaying '2058' instead of '1958'
df_raw.loc[df_raw["dob"] > "2015-01-01", "dob"] = df_raw["dob"].apply(lambda x: x - pd.DateOffset(years=100))

#Given screening date and date of birth, computing age of defendent on date of screening
df_two_years['Age'] = (df_two_years['compas_screening_date'] - df_two_years['dob']).dt.days
df_two_years['age_in_years']= (df_two_years['Age'])/365

#left merge of the two dataframes, as to make sure all rows of the two_years are kept and match to any available row of raw
merged_df = pd.merge(df_two_years,df_raw, how="left", on=["first","last","dob"])

#some rows had no marital status (most likely resulting because a matching row wasn't found in df_raw during merge)
merged_df = merged_df.dropna(subset=["MaritalStatus"])

#changing the type of these columns of data from object to string (needed in order to encode)
merged_df["MaritalStatus"] = merged_df["MaritalStatus"].astype(str)

#Ordinal encoder for labels that have string categories
le = preprocessing.LabelEncoder()
merged_df["sex"] = le.fit_transform(merged_df["sex"])
merged_df["race"] = le.fit_transform(merged_df["race"])
merged_df["MaritalStatus"] = le.fit_transform(merged_df["MaritalStatus"])
merged_df["c_charge_degree"] = le.fit_transform(merged_df["c_charge_degree"])

#combine the two binary targets into one target with three classes
# 0: no reoffense, 1: reoffense, 2: violent reoffense
merged_df['recid_class'] = merged_df['is_recid'] + merged_df['is_violent_recid']

#Feature scaling
merged_df['age_in_years'] = preprocessing.scale(merged_df['age_in_years'])

# Version of dataframe without feature engineering of priors
df_final = merged_df

#subtracting the mean for instances of zero in columns of priors
merged_df['juv_fel_count'] = merged_df['juv_fel_count'].replace(0, -2*(merged_df['juv_fel_count'].mean()))
merged_df['juv_misd_count'] = merged_df['juv_misd_count'].replace(0, -2*(merged_df['juv_misd_count'].mean()))
merged_df['juv_other_count'] = merged_df['juv_other_count'].replace(0, -2*(merged_df['juv_other_count'].mean()))
merged_df['priors_count'] = merged_df['priors_count'].replace(0, -2*(merged_df['priors_count'].mean()))

# Version of dataframe with feature engineering of priors
df_final_priors = merged_df

print('number of samples:', len(merged_df))

"""## Test, Train split for KNN"""

knn_pixie_features_priors = df_final_priors[["sex","age_in_years","race","juv_fel_count","juv_misd_count",
                     "juv_other_count","priors_count","c_charge_degree", "MaritalStatus"]]

knn_pixie_labels_priors = df_final_priors[["recid_class"]]

X_train, X_test , y_train , y_test = train_test_split(knn_pixie_features_priors, knn_pixie_labels_priors, test_size=0.1, random_state=42)

"""## Custom Dataset for Pytorch"""

class PixieDataset (Dataset):
  def __init__(self, training = True, train_prct = 0.8, feat_eng = True):
    self.feat_eng = feat_eng
    if self.feat_eng: tmp_df = df_final_priors 
    else: tmp_df = df_final

    train_df = tmp_df.sample(frac=train_prct, random_state=42) 
    test_df = tmp_df.drop(train_df.index)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if training:
        self.df_pixie_nn = train_df
    else:
        self.df_pixie_nn = test_df

  def __len__(self):
    return len(self.df_pixie_nn)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    pixie_features = self.df_pixie_nn[['sex",'"age_in_years","race","juv_fel_count","juv_misd_count",
                     "juv_other_count","priors_count","c_charge_degree", "MaritalStatus"]].iloc[idx]

    pixie_labels = self.df_pixie_nn[['recid_class']].iloc[idx]

    pixie_features = torch.from_numpy(np.array(pixie_features, dtype='float32'))
    pixie_labels = torch.from_numpy(np.array(pixie_labels, dtype='int64'))

    return pixie_features , pixie_labels

  def get_metrics(self):

    if self.feat_eng: self.merged_df = df_final_priors 
    else: self.merged_df = df_final
    size_of_merged = len(self.merged_df)
    num_non_recid = len(self.merged_df[(self.merged_df['recid_class'] == 0)])
    prct_non_recid = num_non_recid/size_of_merged
    num_recid = len(self.merged_df[(self.merged_df['recid_class'] == 1)])
    prct_recid = num_recid/size_of_merged
    num_violent_recid = len(self.merged_df[(self.merged_df['recid_class'] == 2)])
    prct_violent_recid = num_violent_recid/size_of_merged

    print('\nNon Recid: ', num_non_recid, 'out of ', size_of_merged, ' (', prct_non_recid*100, '%)')
    print('Recid: ', num_recid, 'out of ', size_of_merged, ' (', prct_recid*100, '%)')
    print('Violent Recid: ', num_violent_recid, 'out of ', size_of_merged, ' (', prct_violent_recid*100, '%)')

    size_of_merged_male = len(self.merged_df[(self.merged_df['sex'] == 0)])
    num_non_recid_male = len(self.merged_df[(self.merged_df['recid_class'] == 0)&(self.merged_df['sex'] == 1)])
    prct_non_recid_male = num_non_recid_male/size_of_merged_male
    num_recid_male = len(self.merged_df[(self.merged_df['recid_class'] == 1)&(self.merged_df['sex'] == 1)])
    prct_recid_male = num_recid_male/size_of_merged_male
    num_violent_recid_male = len(self.merged_df[(self.merged_df['recid_class'] == 2)&(self.merged_df['sex'] == 1)])
    prct_violent_recid_male = num_violent_recid_male/size_of_merged_male

    print('\nNon Recid (Males): ', num_non_recid_male, 'out of ', size_of_merged_male, ' (', prct_non_recid_male*100, '%)')
    print('Recid (Males): ', num_recid_male, 'out of ', size_of_merged_male, ' (', prct_recid_male*100, '%)')
    print('Violent Recid (Males): ', num_violent_recid_male, 'out of ', size_of_merged_male, ' (', prct_violent_recid_male*100, '%)')

    size_of_merged_female = len(self.merged_df[(self.merged_df['sex'] == 0)])
    num_non_recid_female = len(self.merged_df[(self.merged_df['recid_class'] == 0)&(self.merged_df['sex'] == 0)])
    prct_non_recid_female = num_non_recid_female/size_of_merged_female
    num_recid_female = len(self.merged_df[(self.merged_df['recid_class'] == 1)&(self.merged_df['sex'] == 0)])
    prct_recid_female = num_recid_female/size_of_merged_female
    num_violent_recid_female = len(self.merged_df[(self.merged_df['recid_class'] == 2)&(self.merged_df['sex'] == 0)])
    prct_violent_recid_female = num_violent_recid_female/size_of_merged_female

    print('\nNon Recid (Females): ', num_non_recid_female, 'out of ', size_of_merged_female, ' (', prct_non_recid_female*100, '%)')
    print('Recid (Females): ', num_recid_female, 'out of ', size_of_merged_female, ' (', prct_recid_female*100, '%)')
    print('Violent Recid (Females): ', num_violent_recid_female, 'out of ', size_of_merged_female, ' (', prct_violent_recid_female*100, '%)')

#Create Datasets for training and testing
Pixie_NN_train = PixieDataset(training=True, train_prct = 0.9, feat_eng=False)
Pixie_NN_test = PixieDataset(training=False, train_prct = 0.9, feat_eng=False)
Pixie_NN_train_priors = PixieDataset(training=True, train_prct = 0.9, feat_eng=True)
Pixie_NN_test_priors = PixieDataset(training=False, train_prct = 0.9, feat_eng=True)
print('Pixie NN Training Dataset initialized (num of samples :', Pixie_NN_train.__len__(), ')\n')
print('Pixie NN Testing Dataset initialized (num of samples :', Pixie_NN_test.__len__(), ')\n')

"""# Dataset Metrics"""

Pixie_NN_train.get_metrics()