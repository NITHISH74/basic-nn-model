# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model
![image](https://user-images.githubusercontent.com/94164665/225720954-d6f42f19-4ece-424a-9620-8298feb68723.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```
### Importing Modules
from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

### Authenticate &  Create Dataframe using Data in Sheets
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('Multiple').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Table':'int'})
df = df.astype({'Product':'int'})

### Assign X and Y values
x = df[["Table"]] .values
y = df[["Product"]].values

### Normalize the values & Split the data
scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

### Create a Neural Network & Train it
ai = Seq([
    Den(8,activation = 'relu',input_shape=[1]),
    Den(15,activation = 'relu'),
    Den(1),
])

ai.compile(optimizer = 'rmsprop',loss = 'mse')

ai.fit(x_train,y_train,epochs=2000)
ai.fit(x_train,y_train,epochs=2000)

### Plot the Loss
loss_plot = pd.DataFrame(ai.history.history)
loss_plot.plot()

### Evaluate the model
err = rmse()
preds = ai.predict(x_test)
err(y_test,preds)

### Predict for some value
x_n1 = [[30]]
x_n_n = scaler.transform(x_n1)
ai.predict(x_n_n)
```
## Dataset Information

![image](https://user-images.githubusercontent.com/94164665/225721133-2290050d-1d38-4203-ba1f-9413fd76432f.png)

## OUTPUT
### Training Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/94164665/225721286-29e0d619-dcf9-4537-92b4-bc4f63474701.png)


### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/94164665/225721353-bbe5c669-c8ff-41f0-87e3-244358487de0.png)

### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/94164665/225721414-db9aa0a1-64f6-4d21-a1d1-9872c9af2256.png)


## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully

