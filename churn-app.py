# Last amended: 02nd October, 2021
#               Gandhi Jayanti
# Myfolder: 1/home/ashok/Documents/churnapp
#           VM: lubuntu_healthcare
# Ref: https://builtin.com/machine-learning/streamlit-tutorial
#
# Objective:
#             Deploy an ML model on web
#
########################
# Notes:
#       1, Run this app in its folder, as:
#          cd /home/ashok/Documents/churnapp
#          streamlit  run  churn-app.py
#       2. Accomanying file to experiment is
#          expt.py
########################

# 1.0 Call libraries
# Install as: pip install streamlit
# Better create a separate conda environment for it
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

#import seaborn as sns
#import matplotlib.pyplot as plt


# 1.1 Set pandas options. None means no truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Write some body-text on the Web-Page:

st.write("""
# Churn Prediction App

Customer churn is defined as the loss of customers after a certain period of time.
Companies are interested in targeting customers who are likely to churn. They can
target these customers with special deals and promotions to influence them to stay
with the company.

This app predicts the probability of a customer churning using Telco Customer data. Here
customer churn means the customer does not make another purchase after a period of time.
""")


# 2.0 Read data from current folder
#     Default folder is where streamlit
#     is being run. So this file
#     should be in /home/ashok/Documents/churnapp
#     Else, home folder is the default.
df_selected = pd.read_csv("telco_churn.csv")

# 2.1 We will select only a few columns
#     for our model:
cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'tenure', 'MonthlyCharges', 'Churn']
df_selected_all = df_selected[cols].copy()

# 3.0 We will create a file download link
#     in our webapp
def filedownload(df):
    csv = df.to_csv(index=False)  # csv is now a string
    csv = csv.encode()            # csv is b' string or bytes
    b64 = base64.b64encode(csv)   # b64 is base64 encoded binary
    b64 = b64.decode()            # b64 is decoded to one of 64 characters
    # 3.1 Create an html link to download datafile
    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
    # 3.2 Return href object
    return href


#st.set_option('deprecation.showPyplotGlobalUse', False)

# 3.3 Finally display the href link
href = filedownload(df_selected_all)
st.markdown(href, unsafe_allow_html=True)

# 4.0 Create a component to upload data file in the sidebar:

uploaded_file = st.sidebar.file_uploader(
                                          "Upload your input CSV file",
                                           type=["csv"]
                                         )

# 4.1 Read data fro file. Else, read from widgets
if uploaded_file is not None:
    # 4.2 Read the uploaded file
    input_df = pd.read_csv(uploaded_file)
else:
    # 4.3 Define a function to create data from widgets
    def user_input_features():
        # 4.4 Create four widgets
        gender         = st.sidebar.selectbox('gender',('Male','Female'))
        PaymentMethod  = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
        MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0,118.0, 18.0)
        tenure         = st.sidebar.slider('tenure', 0.0,72.0, 0.0)
        # 4.5 Collect widget data in a dictionary
        data = {
                'gender':        [gender],
                'PaymentMethod': [PaymentMethod],
                'MonthlyCharges':[MonthlyCharges],
                'tenure':        [tenure]
                }
        # 4,6 Transform data to DataFrame
        features = pd.DataFrame(data)
        # 4.7 Return dataframe of features
        return features
    # 4.8 Call the function and get a 1-row DataFrame
    input_df = user_input_features()


# 5.0 To fill NA values, we may import
#     our original DataFrame
churn_raw = pd.read_csv('telco_churn.csv')
# 5.1 Firt fill up NAs in it
churn_raw.fillna(0, inplace=True)
churn = churn_raw.drop(columns=['Churn'])

# 5.2 Stack vertically 1-row data amd
#     just read DataFrame:
df = pd.concat([input_df,churn],axis=0)
# df

# 5.3 Transform complete data to dummy features
# 5.3.1 Our cat columns
encode = ['gender','PaymentMethod']
for col in encode:
    # 5.3.2 New column names would be: colName_levelName
    dummy = pd.get_dummies(df[col], prefix=col)
    # 5.3.3 Concat these horizontally with existing
    #       DataFrame
    df = pd.concat([df,dummy], axis=1)
    # 5.3.4 Delete the categorical column.
    del df[col]

# 5.4 Just read the first row
df = df[:1]      # Selects only the first row (the user input data)
df.fillna(0, inplace=True)

# 5.5 What are our feature names?
#    Eight features, in all:

features = ['MonthlyCharges',
            'tenure',
            'gender_Female',
            'gender_Male',
            'PaymentMethod_Bank transfer (automatic)',
            'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check'
            ]

# 6.0 Get a data subset of our features
df = df[features]

# 6.1 Displays the user input features
st.subheader('User Input features')
print(df.columns)

# 6.2
if uploaded_file is not None:
    # 6.2.1 Write the first row
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# 7.0 Read in saved classification model
#     from the current folder:
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))

# 7.1 Apply model to make predictions
prediction = load_clf.predict(df)               # 'prediction' will have value of 1 or 0
prediction_proba = load_clf.predict_proba(df)   # Prediction probability

# type(prediction)          # numpy.ndarray
# type(prediction_proba)    # numpy.ndarray

# 8.0 Display Labels
st.subheader('Prediction')
churn_labels = np.array(['No','Yes'])  # churn_labels is an array of strings
                                       # churn_labels[0] is 'No' and churn_labels[1] is 'Yes'
st.write(churn_labels[prediction])     # Display 'Yes' or 'No' depending upon value of
                                       # 'prediction'

# 8.1 Also display probabilities
st.subheader('Prediction Probability')
# 8.2 Numpy arrays are displayed with column names
#     as 1 or 0
st.write(prediction_proba)
######################################
