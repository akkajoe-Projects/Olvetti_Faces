import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

features_df=pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/olivetti_X.csv',header=None)
print('features_df', features_df.head())

target_df=pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/olivetti_y.csv',header=None)
print('target_df', target_df.head())

print(f'The number of rows of the Features DataFrame are {fdf.shape[0]} and the number of columns of the Features DataFrame are {fdf.shape[1]} ')
print(f'The number of rows of the Target DataFrame are {tdf.shape[0]} and the number of columns of the Target DataFrame are {tdf.shape[1]} ')

# Merge the Feature and Target DataFrame
df= features_df[features_df.columns]
df['target']=target_df[0]
print('shape of df', df.shape)
print('dataframe', df.head())

# Checking the distribution of the labels in the target column.
print("df['target'].value_counts(): ", df['target'].value_counts())
print("df['target'].unique()", df['target'].unique())

# Creating a Python function to visualise exactly one sample image of a label that exists in the 'df' DataFrame.
group_df=df.groupby(by='target')
# Define the function to visualise the images
def vis(d):
  # The groups for the input label
  d_df=group_df.get_group(d)
  # The row number of the first instance in the group
  drownum=d_df.index.values[0]
  # The data of the row number selected 
  dar=df.iloc[drownum,:-1]
  dar=dar.values.reshape(64,64)
  # Reshaping the data into a 2D array of 64 x 64.
  plt.figure(figsize=(16,7),dpi=99)
  # Creating the image
  plt.title(f" image for {d}")
  plt.imshow(dar, cmap='gray')
  plt.show()

# Calling the function to generate the 40 images
for i in range(0,40):
  vis(i)

# Train Test Split
from sklearn.model_selection import train_test_split
features = df.iloc[:,:-1]
target = df['target']
f_train,f_test,t_train,t_test=train_test_split(features,target,test_size=0.30,random_state=42)
print('f_train.shape', f_train.shape)
print('f_test.shape', f_test.shape)
print('t_train.shape', t_train.shape)
print('t_test.shape', t_test.shape)

from sklearn.svm import SVC
svc_model=SVC(kernel='linear')
svc_model.fit(f_train,t_train)
print('Accuracy of SVM Classification Model', svc_model.score(f_train,t_train))
svm_train_pred = svc_model.predict(f_train)
svm_test_pred = svc_model.predict(f_test)

#vModel Evaluation
from sklearn.metrics import confusion_matrix,classification_report
# Creating the training confusion matrix DataFrame
cmdf=pd.DataFrame(confusion_matrix(t_train,svm_train_pred))
# Creating the heatmap
plt.figure(figsize=(16,7),dpi=99)
sns.heatmap(cmdf,cmap='YlGnBu',annot=True,fmt='g')
plt.show()

print('classification_report(t_train,svm_train_pred): ' , classification_report(t_train,svm_train_pred))

# Creating the testing confusion matrix DataFrame
cmdf2=pd.DataFrame(confusion_matrix(t_test,svm_test_pred))
# Creating the heatmap
plt.figure(figsize=(16,7),dpi=99)
sns.heatmap(cmdf2,cmap='icefire',annot=True,fmt='g')
plt.show()

print('classification_report(t_test,svm_test_pred): ', classification_report(t_test,svm_test_pred))