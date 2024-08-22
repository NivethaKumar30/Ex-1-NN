<H3>ENTER YOUR NAME : NIVETHA.K</H3>
<H3>ENTER YOUR REGISTER NO. 212222230102 </H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
Import Libraries
```

from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
Read the dataset
```
df=pd.read_csv("Churn_Modelling.csv")
```
Checking Data
```
df.head()
df.tail()
df.columns
```
Check the missing data
```
df.isnull().sum()


Check for Duplicates

df.duplicated()

Assigning Y

y = df.iloc[:, -1].values
print(y)
```
Check for duplicates
```
df.duplicated()
```
Check for outliers
```
df.describe()
```
Dropping string values data from dataset
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
```
Normalize the dataset
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
Split the dataset
```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
Training and Testing model
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```


## OUTPUT:
Data checking:
![image](https://github.com/user-attachments/assets/d05f34d3-de3d-496a-b6b8-4002d1781900)

Missing Data:

![image](https://github.com/user-attachments/assets/2c801b42-20e5-4d2e-b8b0-e8c9993b6626)


Duplicates identification:

![image](https://github.com/user-attachments/assets/ffbab17c-083a-4602-accc-eaafc141ca74)

Vakues of 'Y':

![image](https://github.com/user-attachments/assets/2f839b2e-e0a5-4f68-adea-636d94a1a85e)

Outliers:

![image](https://github.com/user-attachments/assets/05851ca0-40f7-46b5-94f9-5cba457ddc6f)

Checking datasets after dropping string values data from dataset:

![image](https://github.com/user-attachments/assets/5c7d6569-ff7a-4902-8e40-5d7635d41edc)

Normalize the dataset:

![image](https://github.com/user-attachments/assets/f5593103-b638-41dd-9e41-41e5ad63f988)

Split the dataset:

![image](https://github.com/user-attachments/assets/a152252b-6532-4a40-99b0-f164b7409b43)

Training and Testing model:

![image](https://github.com/user-attachments/assets/51160bce-521e-4a0f-8837-3b6e769bf8a1)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


