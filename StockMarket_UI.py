#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import tensorflow as tf
import yfinance as yf
from textblob import TextBlob
import nltk
import pandas_datareader as data
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import date
import datetime as datetime
import streamlit as st
import time
from dateutil.relativedelta import relativedelta


API="EQ6PLX2EW5FA0DFL"


   
    
def sentiment(Symbol,API,time_from):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": Symbol,
        "apikey": API,
        "time_from":time_from,
        "limit":"200"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data
    
    #Historical data function
def history(Symbol,API):
    url = "https://www.alphavantage.co/query?"
    params = {
       "function":"TIME_SERIES_DAILY_ADJUSTED",
       "symbol":Symbol,
       "apikey":API,
        "outputsize":"full"
    }
    response_HD = requests.get(url, params=params)
    data = response_HD.json()
    return data


with open('C:\\Users\\admin\\Desktop\\BCS 4.2\\Final year project\\Implementation\\Project semi-final\\styles.css') as f:
        css = f.read()

        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


header= st.container()

# Add widgets to the columns
with header:
    intro=st.title("Stock Market Analysis and Prediction ")
          
    #st.markdown( 
              # )
    Symbol=st.sidebar.text_input('Enter the name of your stock',"PG").upper()
    time_from=st.sidebar.text_input('Enter the start date(YYYYMMDDTHHMM)',"20220304T0000")
    start_date = time_from[:4] + "-" + time_from[4:6] + "-" + time_from[6:8]
    # buttons
    with open('C:\\Users\\admin\\Desktop\\BCS 4.2\\Final year project\\Implementation\\Project semi-final\\stockstyle.css') as f: 
         style= f.read()
    headers=["Historical Data", "Chart", "Sentiments","Predict","About us"]
    
    tab1, tab2, tab3,tab4,tab5 = st.tabs(headers)
    

    with tab1:
            
        #Historical Data
            hist_data=history(Symbol,API)
            stock_data = hist_data["Time Series (Daily)"]
            stock_data =pd.DataFrame(stock_data).T
            #stock_data.columns

            stock_data=stock_data.rename(columns={'1. open':"Open","2. high":"High","3. low":"Low","4. close":"Close","6. volume":"Volume"})
            hd_cols=["Open","High","Low","Close","Volume"]
            stock_data=stock_data[hd_cols].astype(float)
            stock_data= stock_data.loc[:start_date,:]
            stock_data.index=pd.to_datetime(stock_data.index,format='%Y-%m-%d')
            #stock_data.tail()
           
            s=stock_data
            s.index=s.index.date
            s=stock_data.style.set_properties(**{'background-color': 'transparent',
                           'color': 'black',
                           'border-color': 'black'}) 
            st.markdown("<p class=parag>Historical data upto the previous day</p> "
                        f"<style>{style}</style>"
                        
                        
            ,unsafe_allow_html=True )               
            with st.spinner('Please wait...'):
                 time.sleep(10)     
            st.dataframe(s,5000,2000)
    with tab2:
     # Create selectbox
            options = ['Open', 'Close', 'High', 'Low']
            selected_option = st.sidebar.selectbox('Select chart to display', options)

            # Filter data based on selected option
            if selected_option == 'Open':
                price_col = stock_data['Open']
                price="Open price"
            elif selected_option == 'Close':
                price_col = stock_data['Close']
                price="Close price"
            elif selected_option == 'High':
                price_col = stock_data['High']
                price="High price"
            else:
                price_col = stock_data['Low']
                price="Low price"
            
        # Plot the stock data
            fig, ax = plt.subplots(figsize=(10,7))
            price_col.plot(ax=ax, color="red", linestyle="-", linewidth=2)

            # Add grid and legend
            plt.grid(True)
            plt.legend([price], loc="upper left")

            # Set the chart title and axis labels
            plt.title(f"{Symbol} Stock Prices")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")

            # Show the chart in Streamlit
            st.pyplot(fig)

    with tab3:
        #SENTIMENT DATA
            data=sentiment(Symbol,API,time_from)
            news_items = data['feed']
            feed=pd.DataFrame(news_items)
            feed["time_published"] = pd.to_datetime(feed["time_published"])

            feed = feed.query('time_published.dt.dayofweek<5')

            feed['time_published'] = feed['time_published'].dt.date
            #feed
            cols=['title','time_published','summary','overall_sentiment_score','overall_sentiment_label']
            sentiment_data=feed[cols]
            #text polarity
            # function to get sentiment score using TextBlob
            def get_sentiment_score(text):
                blob = TextBlob(text)
                return blob.sentiment.polarity

            text=sentiment_data['summary']+" "+sentiment_data['title']
            # add sentiment score column to dataframe
            sentiment_data['Sentiment_Score_2'] = text.apply(get_sentiment_score) 
            #st.header(f"Latest News Headlines for {Symbol}")
            st.markdown(f"<p class=parag>Latest News Headlines for {Symbol}</p> "
                        f"<style>{style}</style>"
                        
            ,unsafe_allow_html=True )
            user_cols=['title','time_published','summary','overall_sentiment_label']
            user_view=sentiment_data[user_cols]
            st.dataframe(user_view)
   
    #st.set_page_config(title="Stock Market Analysis and Prediction ", layout="wide")
    #t.image()  
    with open('C:\\Users\\admin\\Desktop\\BCS 4.2\\Final year project\\Implementation\\Project semi-final\\styles.css') as f:
        css = f.read()

        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    #st.markdown("",unsafe_allow_html=True)
    
contents= st.container()  
with contents:  
            #prepare sentiment input data
            input_col=['time_published','Sentiment_Score_2']
            sentiment_input=sentiment_data[input_col] 
            sentiment_input=sentiment_input.rename(columns={'time_published':'Date'})
            #sentiment_input
            grouped_sent=sentiment_input.groupby(['Date']).mean().reset_index()
            grouped_sent["Date"]=pd.to_datetime(grouped_sent["Date"],format="%Y-%m-%d")
            grouped_sent["Date"]=grouped_sent["Date"].dt.date
            score_col=['Sentiment_Score_2']
            grouped_sent[score_col]=grouped_sent[score_col].astype(float)
            #grouped_sent
            
            #Combine sentiment data with historical data
            stock_data=stock_data.reset_index()
            stock_data=stock_data.rename(columns={'index':'Date'})
            input_data= pd.merge(grouped_sent,stock_data, on='Date', how='inner')
            #input_data
            input_data.dropna(axis=0,inplace=True)
            #input_data 

         #TECHNICAL DATA
            #calculate exponential Moving Averages
            input_data['EMA_50']=input_data['Close'].ewm(span=50,adjust=False).mean()
            input_data['EMA_200']=input_data['Close'].ewm(span=200,adjust=False).mean() 

            #calculate trend where 1 represents rising and 0 represents falling
            def get_trend(row):
                if row['EMA_50'] > row['EMA_200'] and row['Open'] > row['EMA_50'] and row['Close'] > row['EMA_50']:
                    return 1
                else:
                    return 0

            input_data['Trend'] = input_data.apply(get_trend, axis=1)
            #drop any null rows
            input_data.dropna(inplace=True)      
            #input_data
            

         #FUNDAMENTAL DATA  
            #Income Statement quarterly
            url = 'https://www.alphavantage.co/query'
            params = {
                "function": "INCOME_STATEMENT",
                "symbol": Symbol,
                "apikey": API
            }
            response = requests.get(url, params=params)
            data_inc = response.json()
            #print(data_inc.keys()) 
            income_data=pd.DataFrame(data_inc['quarterlyReports'])
            #income_data.head()            
            
            #Balance Sheet
            url = 'https://www.alphavantage.co/query'
            params = {
                "function": "BALANCE_SHEET",
                "symbol": Symbol,
                "apikey": API
            }
            response = requests.get(url, params=params)
            data_bals = response.json()
            #print(data_bals.keys())  
            balance_sheet=pd.DataFrame(data_bals['quarterlyReports'])
            #balance_sheet.head()  

            #income staement columns
            income_cols=['fiscalDateEnding','netIncome']
            inc_df=income_data[income_cols]
            net_income=['netIncome']
            inc_df[net_income]=inc_df[net_income].astype(float)
            inc_df['fiscalDateEnding']=pd.to_datetime(inc_df['fiscalDateEnding'],format='%Y-%m-%d')
            #inc_df

            #balancesheet columns
            bals_cols=['fiscalDateEnding','totalShareholderEquity','commonStockSharesOutstanding']
            bals_df=balance_sheet[bals_cols]
            share_cols=['totalShareholderEquity','commonStockSharesOutstanding']
            bals_df['fiscalDateEnding']=pd.to_datetime(bals_df['fiscalDateEnding'],format='%Y-%m-%d')
            bals_df[share_cols]=bals_df[share_cols].astype(float)
            #bals_df 
            
            fd_data=pd.merge(bals_df,inc_df,on='fiscalDateEnding',how='inner')
            
            #fd_data.columns
            
            # Calculate the financial ratios
            fd_data["ROE"] = fd_data["netIncome"]/fd_data["totalShareholderEquity"]
            fd_data["EPS"] = fd_data["netIncome"]/fd_data["commonStockSharesOutstanding"]
            fd_data["P/E"] = stock_data["Close"]/fd_data["EPS"]

            # Select the desired columns
            selected_columns = ["fiscalDateEnding" ,"ROE", "EPS", "P/E"]
            fd_data['Quarter'] = pd.PeriodIndex(fd_data['fiscalDateEnding'], freq='Q')
            fd_data = fd_data[selected_columns]
            #create an empty row to hold first row values after shift
            fd_data.loc[-1] = [None] * len(fd_data.columns)
            fd_data.index =fd_data.index + 1
            fd_data = fd_data.sort_index()
            #shift rows
            fd_data[['ROE', 'EPS', 'P/E']] = fd_data[['ROE', 'EPS', 'P/E']].shift(-1)
            #fd_data
            #assign next quarter date to first row of first column
            new_date = fd_data.loc[1,"fiscalDateEnding"] + relativedelta(months=3)
            fd_data.loc[0,"fiscalDateEnding"]= new_date
            #add a column quarter
            fd_data['Quarter'] = pd.PeriodIndex(fd_data['fiscalDateEnding'], freq='Q')
            #fd_data=fd_data.dropna()
            
            
            #Add fundamental data to our input data
            
            #Add quarter column to input data
            input_data['Quarter'] = pd.PeriodIndex(input_data['Date'], freq='Q')
            #input_data
            final_input=pd.merge(input_data,fd_data,on='Quarter',how='left')
            symbol_label = {
                            'DOW': 0,
                            'GOOG': 1,
                            'AMZN': 2,
                            'MSFT':3,
                            'AAPL':4,
                            'TSLA':5,
                            'PG':6, 
                            'META':7, 
                            'AMD':8,
                            'NFLX':9, 
                            'TSM':10, 
                            'KO':11, 
                            'F':12, 
                            'COST':13, 
                            'DIS':14, 
                            'VZ':15, 
                            'CRM':16, 
                            'INTC':17, 
                            'BA':18,
                            'BX':19, 
                            'NOC':20, 
                            'PYPL':21, 
                            'ENPH':22, 
                            'NIO':23, 
                            'ZS':24, 
                            'XPEV':25
                        }

            symbol_vec = symbol_label[Symbol]
            final_input['symbol_label'] = symbol_vec
            final_input.sort_values(by='Date', ascending = False, inplace = True)
            final_input=final_input.reset_index()
            #final_input
            
with tab4:
    st.write("Click here to get your predictions")
        
    predictbtn=st.button('Predict')               
    if predictbtn:
        with st.spinner('Analyzing data...'):
                 time.sleep(10)
        

        from tensorflow.keras.models import *
        stock_model = "C:\\Users\\admin\\Desktop\\BCS 4.2\\Final year project\\Implementation\\Project semi-final\\Stock_prediction_Timeseries final.h5"
        #final_input=pd.read_csv("C:\\Users\\admin\\Desktop\\BCS 4.2\\Final year project\\Implementation\\AMZN_final_input.csv")
        cols = ['Open', 'Close', 'Sentiment_Score_2', 'EMA_50','EMA_200','Trend', 'ROE', 'EPS', 'P/E' ,'symbol_label','High', 'Low']
        model_input= final_input[cols]
        model_input
                
        #normalize test data
        scaler = MinMaxScaler(feature_range=(0, 1))
        model_data_scaled=scaler.fit_transform(model_input)

        y_test_scaled=[]
        x_test_scaled=[]
        #Declare input and target test lists
        for i in range(5,len(model_input)-1):
            x_test_scaled.append(model_data_scaled[i-5:i,:])
            y_test_scaled.append(model_data_scaled[i+1,[10,11]])


        #convert to arrays
        y_test_scaled,x_test_scaled=np.array(y_test_scaled),np.array(x_test_scaled)
        #y_test_scaled

        model = load_model(stock_model)
        prediction_scaled=model.predict(x_test_scaled)

        # create new scaler object with n_features = 2
        scaler_new = MinMaxScaler(feature_range=(0, 1))
        scaler_new.n_features_in_ = 2
        scaler_new.min_, scaler_new.scale_ = scaler.min_[[10,11]], scaler.scale_[[10,11]]

        # inverse transform predictions
        prediction = scaler_new.inverse_transform(prediction_scaled)
        prediction_dic={'Predicted High':prediction[:,0],'Predicted Low':prediction[:,1]}
        prediction_df=pd.DataFrame(prediction_dic)  
        #prediction_df.iloc[0,:]        
         

        st.markdown(f"<p class=val>Todays High will be : {prediction_df.iloc[0,0]}</p> "
                    f"<p class=val>Todays Low will be : {prediction_df.iloc[0,1]}</p> "
                        f"<style>{style}</style>"
                        
            ,unsafe_allow_html=True )



