# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('DLExp1').sheet1
data = worksheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

X = df[['Input']].values
y = df[['Output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1

ai=Sequential([
    Dense(7,activation='relu'),
    Dense(6,activation='relu'),
    Dense(1)
])
ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(X_train1,y_train,epochs=2000)
ai.fit(X_train1,y_train,epochs=2000)

## Plot the loss
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

## Evaluate the model
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)

# Prediction
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```

## Dataset Information 

![Screenshot 2023-08-19 104735](https://github.com/sarveshjustin/basic-nn-model/assets/113497481/5431e236-db62-4583-987c-ca750424fede)


## OUTPUT

![Screenshot 2023-08-19 104806](https://github.com/sarveshjustin/basic-nn-model/assets/113497481/f19e8230-5aa0-46fb-a5aa-7f908f13d3e9)
### Test Data Root Mean Squared Error

![Screenshot 2023-08-19 104854](https://github.com/sarveshjustin/basic-nn-model/assets/113497481/a9e2420c-3027-47e2-963d-b3c9d9762ff3)


### New Sample Data Prediction

![Screenshot 2023-08-19 104917](https://github.com/sarveshjustin/basic-nn-model/assets/113497481/d10284ed-4c92-4f06-82ab-a394808b754d)


## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully.
