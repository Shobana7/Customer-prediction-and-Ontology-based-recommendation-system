import pandas as pd  
import newcombine

def givedata():    
    df=pd.read_csv("cartesianprod.csv")
    del df['Unnamed: 0']
    df=df.sample(n = 1)

    df.to_csv('TestInput.csv')
    
    Name=df.iloc[0].Name
    ID=df.iloc[0].ID
    AvgPrice=df.iloc[0].AvgPrice
    MinPrice=df.iloc[0].MinPrice
    MaxPrice=df.iloc[0].MaxPrice

    return Name,ID,AvgPrice,MinPrice,MaxPrice

def getsamedata():
    df=pd.read_csv("TestInput.csv")
    del df['Unnamed: 0']
    Name=df.iloc[0].Name
    ID=df.iloc[0].ID
    AvgPrice=df.iloc[0].AvgPrice
    MinPrice=df.iloc[0].MinPrice
    MaxPrice=df.iloc[0].MaxPrice

    return Name,ID,AvgPrice,MinPrice,MaxPrice

def get_lstm_output(lstm_model):
    lstmOutput =newcombine.run_lstm_for_input(lstm_model)
    lstmOutput=lstmOutput.flatten()
    if ( lstmOutput ==0 ):
        return 0
    else :
        return 1

def get_rf_output(rf_model):
    rfOutput =newcombine.run_rf_for_input(rf_model)
    rfOutput=rfOutput.flatten()
    print("RF OUTPUT IS ")
    print(rfOutput) 
    if ( rfOutput ==0 ):
        return 0
    else :
        return 1



# def train_models():
#     lstm_model=newcombine.train_lstm()
#     rf_model=newcombine.train_rf()

#     return lstm_model,rf_model