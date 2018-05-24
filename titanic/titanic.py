'''
Thanks to this tutorial for providing a nice framework for me to build my model on: https://www.kaggle.com/linxinzhe/tensorflow-deep-learning-to-solve-titanic

Author: Christoffer Olsson, 2018
'''


import pandas as pd
import tensorflow as tf
import numpy as np

train_path = 'train.csv'
test_path = 'test.csv'

train_data = pd.read_csv(train_path) #load files to memory
test_data = pd.read_csv(test_path)

##################################################################################################################
######### Feature Engineering! ###################################################################################
##################################################################################################################

from sklearn.preprocessing import Imputer #This inputs values into dataframes

def nan_padding(data,columns):
    '''
    If some row has 'NaN' in it we can insert values using this function
    '''
    for column in columns:
        imputer = Imputer()
        data[column] = imputer.fit_transform(data[column].values.reshape(-1,1)) #note this function takes arguments so that different values can be inserted
    return data

nan_columns = ['Age', 'SibSp', 'Parch']


train_data = nan_padding(train_data, nan_columns)
test_data = nan_padding(test_data, nan_columns)


#save passenger id, the index corresponds to the id of the passenger in this case
test_passenger_id=test_data["PassengerId"]

def drop_not_concerned(data, columns):
    '''
    This function drops columns we are not interested in!
    '''
    return data.drop(columns, axis=1)


columns_we_dont_care_about = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]

train_data = drop_not_concerned(train_data, columns_we_dont_care_about)
test_data = drop_not_concerned(test_data, columns_we_dont_care_about)

def one_hot_encode_some_integers(data, columns):
    '''
    Turn a column were values are 1,2,3 into three columns of either 0 or 1
    '''
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1) #this line turns a column of [1,3] into three columns of either 0 or 1
        data = data.drop(column, axis=1) #remove the column with scalar values
    return data


dummy_columns = ['Pclass']
train_data = one_hot_encode_some_integers(train_data, dummy_columns)
test_data = one_hot_encode_some_integers(test_data, dummy_columns)

from sklearn.preprocessing import LabelEncoder
def sex_as_string_to_sex_as_integer(data):
    '''
    Encode Male and Female into 0 or 1
    '''
    label_encoder = LabelEncoder()
    label_encoder.fit(['male', 'female'])
    data["Sex"]=label_encoder.transform(data["Sex"])
    return data


train_data = sex_as_string_to_sex_as_integer(train_data)
test_data = sex_as_string_to_sex_as_integer(test_data)


from sklearn.preprocessing import MinMaxScaler

def normalize_age(data):
    '''
    This function takes age and normalizes it into the range [0,1]
    '''
    scaler = MinMaxScaler()
    data['Age'] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data


train_data = normalize_age(train_data)
test_data = normalize_age(test_data)


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def split_valid_test_data(data, fraction=(1 - 0.8)):
    '''
    Split the training data into a training set, and a validation set!
    '''
    data_y = data["Survived"] #the ground truth values.
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1) #remove the column of truth from the training data.

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction) #these functions are amazing

    return train_x.values, train_y, valid_x, valid_y


train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data) #data that is ready to go!


##################################################################################################################
########################################### MODEL!! ##############################################################
##################################################################################################################


