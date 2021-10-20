#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[3]:


import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Suppress gast errors unresolved by developers when using tensorflow version >=2.0.
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

from tkinter import *


# ### First GUI Window

# In[4]:


root = Tk()
root.geometry("800x350") 
root.resizable(0, 0)

Label(root, text = "-----------------WELCOME!  TO USE THE STOCK PRICE PREDICTOR PLEASE FOLLOW THE STEPS BELOW:-------------------").grid(row = 0, sticky = W)
Label(root, text = " ").grid(row = 1, sticky = W)
Label(root, text = "1. Enter a company's ticker that is listed on the NASDAQ / NYSE. (Company must have been listed for 2 years):").grid(row = 2, sticky = W)
Label(root, text = "2a. Enter a date, that the stock market will be open, then click the 'Enter Date' button to confirm. (Format: Year-Month-Day)").grid(row = 4, sticky = W)
Label(root, text = " ").grid(row = 6, sticky = W)
Label(root, text = "2b. Alternatively, click the 'Next Market Day' button for the next trading day's prediction.").grid(row = 7, sticky = W)
Label(root, text = " ").grid(row = 8, sticky = W)
Label(root, text = "3. Finally click 'Predict!' to start or 'Cancel' to quit.").grid(row = 9, sticky = W)
Label(root, text = "--------------------------------------------------------------------------------------------------------------------------------").grid(row = 11, sticky = W)
Label(root, text = f"THIS IS NOT FINANCIAL ADVICE.").grid(row=12, sticky=W)
Label(root, text = f"ANY LOSSES INCURED FROM TRADING BASED ON DIRECTION REUSLTS ARE SOLELY YOUR RESPONSIBILITY.").grid(row=13, sticky=W)

frame = Frame(root, bg="#263D42")
frame.place(relwidth=1, relheight=1, relx=0, rely=0.9)

frame2 = Frame(root, bg="#263D42")
frame2.place(relwidth=1, relheight=1, relx=0.95, rely=0)

userInput = Entry(root)
userInput.grid(row = 3, column = 0, sticky = W)

dateInput = Entry(root)
dateInput.grid(row = 5, column = 0, sticky = W)

def quit():
    root.destroy()
    
    global company
    company = ""
    sys.exit(0)
    
def getInput():
    inp = userInput.get()
    root.destroy()

    global company
    company = inp
    
def getDate():
    date = dateInput.get()
    date = dt.datetime.strptime(date, '%Y-%m-%d')
    
    global endDate
    endDate = date
    
def tomorrow():
    date = dt.date.today() + dt.timedelta(days=1)

    global endDate
    endDate = date


Button(root, text = "Enter Date", command = getDate).grid(row = 5, column = 1, sticky = W)
Button(root, text = "Next Market Day.", command = tomorrow).grid(row = 7, column = 1, sticky = W)
Button(root, text = "Predict!", command = getInput).grid(row = 9, column = 1, sticky = W)
Button(root, text = "Cancel (EXIT)", command = quit).grid(row = 10, column = 1, sticky = W)

mainloop()


# ### import data for training from yfinance

# In[5]:


company = company.upper()
start = '2016-01-01'
end = '2020-01-01'

data = yf.download(company, start, end);

# Exit if no ticker data is found.
if len(data) == 0:
    print("ERROR - TICKER NOT FOUND")
    sys.exit(0)


# ### Scale data

# In[6]:


# Scale data
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 30    # how many days to look back, for next day predicted

# training data lists
x_train = [] 
y_train = []

# append the amount of prior days to x_train, next day to y_train.
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days: x, 0])
    y_train.append(scaled_data[x , 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# ### Create the Dense LSTM Model and train it on the training data

# In[7]:


model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Dropout(0.2))

# Prediction of the next closing day's value
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit the model, choose how many repetitions (Epochs).
model.fit(x_train, y_train, epochs = 10, batch_size = 64)


# ### Test the model accuracy on Existing Data

# In[8]:


test_start = dt.datetime(2020, 1, 1)
test_end = endDate                                # Changed by the user input.

test_data = yf.download(company, test_start, test_end);
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)


# ### Make predictions on Test data

# In[9]:


x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


# ## Predict the next day

# In[10]:


real_data = [model_inputs[len(model_inputs) - prediction_days: len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)


# ## Results

# In[11]:


results = pd.DataFrame()
results['Actual Close Price'] = actual_prices
results['Predicted Close Price'] = predicted_prices

prior_day = results["Actual Close Price"].tail(1)
prior_day = prior_day.values

prior = ["%.4f" % i for i in prior_day]
next_day = ["%.4f" % i for i in prediction] 

test_end -= dt.timedelta(days=1)    # subtract by one day as the test_end in yfinance
                                     # imports up to but not including that date so for yesterdays date subtract

test_end = test_end.strftime('%Y-%m-%d')
endDate = endDate.strftime('%Y-%m-%d')

print("\n\n\n\n\n\n\n\n\n\n\n\n")
print("---------------------------------------------------------------------------------------")
print(f"{company}'s prior day price was: {prior[0]} on the '{test_end}'.")    
print(f"{company}'s next day price is predicted to be: {next_day[0]} on the '{endDate}'/ Next Market Day If Closed.")


# ### Output choice, based on price change above/below small margin

# In[12]:


print("\n")
print("LSTM PREDICTION SAYS:")

choice = "hold"
output = ""

if (prior_day * 1.01) < prediction:
    print(f"--> {company} will go up, action is buy.")
    choice = "buy"
    output = f"{company} will go up, action is buy."
    
elif (prior_day * 0.99) > prediction:
    print(f"--> {company} will go down, action is sell.")
    choice = "sell"
    output = f"{company} will go down, action is sell."
    
else:
    print(f"--> {company} will not move significantly, action is hold/wait.")
    choice = "hold"
    output = f"{company} will not move significantly, action is hold/wait."

print("---------------------------------------------------------------------------------------")


# ### Convert prediction to string, write said prediction + info to a .csv

# In[13]:


def npToString(numpyArray):
    df = pd.DataFrame(numpyArray)
    df = df.rename(columns={0: "Prediction"})
    df = df['Prediction'].values
    pred =["%.4f" % i for i in df]    # converts a numpy array into string(s)
    
    return pred

pred = npToString(prediction)


# In[14]:


def addToFile(file, what):
    f = open(file, 'a').write(what) 


# In[15]:


addToFile("./PREDICTION LOG.csv", f"{company} -> {prediction_days} Days Predicted: " + pred[0] + 
         f" ({choice})" +f" for {endDate}." + "\n")


# ### Second GUI Window, plot results of LSTM, write .png to folder

# In[16]:


root = Tk() 
root.geometry("850x850") 
root.resizable(0, 0)

frame = Frame(root, bg="#263D42")
frame.place(relwidth=1, relheight=1, relx=0, rely=0.90)

frame2 = Frame(root, bg="#263D42")
frame2.place(relwidth=1, relheight=1, relx=0.95, rely=0)

Label(root, text = f"LSTM MODEL PREDICTION FOR '{endDate}'/ Next Market Day If Closed:   -------->   ${pred[0]}").grid(row = 0, sticky = W)
Label(root, text = f"{output.upper()}").grid(row = 1, sticky = W)
Label(root, text = f"-----------------------------------------------------------------").grid(row=2, sticky=W)
Label(root, text = f"CLICK BELOW TO GENERATE A GRAPH OF THE LSTM MODEL'S PREDICTIONS ON THE DATASET.").grid(row = 3, column = 0, sticky = W)
Label(root, text = f"").grid(row=6, sticky=W)
Label(root, text = f"").grid(row=7, sticky=W)
Label(root, text = f"THIS IS NOT FINANCIAL ADVICE.").grid(row=9, sticky=W)
Label(root, text = f"ANY LOSSES INCURED FROM TRADING BASED ON DIRECTION REUSLTS ARE SOLELY YOUR RESPONSIBILITY.").grid(row=10, sticky=W)
Label(root, text = f" ").grid(row=11, sticky=W)

def exit():
    root.destroy()
    
    
def plot():
    fig = plt.Figure(figsize = (10,6))
    a = fig.add_subplot(111)
    
    fig.set_size_inches(5, 5, forward=True)
    
    
    a.plot(actual_prices, color = 'black', label = f'Actual {company} Price')
    a.plot(predicted_prices, color = 'green', label = f'Predicted {company} Price')
    a.set_title(f'{company} Stock Price VS Predicted Price (LSTM)')
    a.set_xlabel('Time (Days)')
    a.set_ylabel(f'{company} Stock Price ($)')
    a.legend()
    a.grid()
    
    canv = FigureCanvasTkAgg(fig, master = root)
    canv.draw()
    
    get_widz = canv.get_tk_widget()
    get_widz.grid(row=5, column =0, sticky=W)
    

    # define the name of the directory to be created
    path = "./figures"

    try:
        os.mkdir(path)
    except OSError:
        print (f"Creation of the directory {path} failed, check if it has already been created.")
    
    
    fig.savefig(f'./figures/{company} results.png', bbox_inches='tight')
    
    Label(root, text = f"PLOT HAS BEEN STORED AS {company} results.png in {path}.").grid(row=6, sticky=W)
    


Button(root, text = "PLOT", command = plot).grid(row = 4, column = 0, sticky = W)
Button(root, text = "EXIT", command = exit).grid(row = 12, column = 0, sticky = W)

mainloop()

