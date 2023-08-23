# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## Theory
We create a simple dataset with one input and one output. This data is then divided into testing and training sets for our Neural Network Model to train and test on. The NN Model contains input layer, 2 nodes/neurons in the hidden layer which is then connected to the final output layer with one node/neuron. The Model is then compiled with an loss function and Optimizer, here we use MSE and rmsprop. 

## Neural Network Model
![Screenshot 2023-08-23 205011](https://github.com/sarveshjustin/basic-nn-model/assets/113497481/8ce730be-da87-4781-9baf-ffc0db8e0f31)



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
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('dataset').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])

df.head()

df=df.astype({'X':'float'})
df=df.astype({'Y':'float'})
df.dtypes

X=df[['X']].values
Y=df[['Y']].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=50)

scaler=MinMaxScaler()
scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)

ai_brain=Sequential([
    Dense(2,activation='relu'),
    Dense(1,activation='relu')
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=X_train_scaled,y=Y_train,epochs=20000)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test_scaled=scaler.transform(X_test)
ai_brain.evaluate(X_test_scaled,Y_test)

prediction_test=int(input("Enter the value to predict: "))
preds=ai_brain.predict(scaler.transform([[prediction_test]]))
print("The prediction for the given input "+str(prediction_test)+" is: "+str(preds))
```

## Dataset Information 
<img width="81" alt="2" src="https://github.com/sarveshjustin/basic-nn-model/assets/113497481/42f42bcf-292a-4135-a4ae-4b3177dd4366">



## OUTPUT
![3](https://github.com/sarveshjustin/basic-nn-model/assets/113497481/5c2e7d85-7029-4e39-967d-77980bb1d2e1)


### Test Data Root Mean Squared Error

<img width="482" alt="4" src="https://github.com/sarveshjustin/basic-nn-model/assets/113497481/c489b645-6e49-43f3-aae4-d09449ae37ed">




### New Sample Data Prediction
<img width="348" alt="5" src="https://github.com/sarveshjustin/basic-nn-model/assets/113497481/5f83e808-315d-427b-924e-b288f49711f3">



## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.
