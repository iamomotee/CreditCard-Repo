import streamlit as st
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


st.markdown("<h1 style = 'color: #191970; text-align: center; font-family: helvetica'>CREDIT CARD DEFAULTERS PREDICTIVE MODEL</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #301934; text-align: center; font-family: cursive '>Built By Future is now</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)

st.image('Credit Card Defaulter.png')

# st.header('Project Background Information', divider = True)
st.write("The primary objective of a credit card defaulters predictive model is to accurately identify individuals or businesses who are at a high risk of defaulting on their credit card payments. By leveraging historical transaction data and customer information, the model aims to predict the likelihood of default before it occurs, enabling financial institutions to take proactive measures to mitigate risk and minimize potential losses.")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)


cs = pd.read_csv('Credit Card Defaulter Prediction.csv')
st.dataframe(cs)

st.sidebar.image('Card User.png', caption = 'Welcome User')

st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# st.sidebar.subheader('Input Variables', divider= True)
age = st.sidebar.selectbox('AGE', cs['AGE'].unique())
bill_amt1 = st.sidebar.number_input('BILL_AMT1', cs['BILL_AMT1'].min(), cs['BILL_AMT1'].max())
pay_amt6 = st.sidebar.number_input('PAY_AMT6', cs['PAY_AMT6'].min(), cs['PAY_AMT6'].max())
bill_amt6 = st.sidebar.number_input('BILL_AMT6', cs['BILL_AMT6'].min(), cs['BILL_AMT6'].max())
bill_amt5 = st.sidebar.number_input('BILL_AMT5', cs['BILL_AMT5'].min(), cs['BILL_AMT5'].max())
pay_amt1 = st.sidebar.number_input('PAY_AMT1', cs['PAY_AMT1'].min(), cs['PAY_AMT1'].max())
bill_amt2 = st.sidebar.number_input('BILL_AMT2', cs['BILL_AMT2'].min(), cs['BILL_AMT2'].max())
pay_amt4 = st.sidebar.number_input('PAY_AMT4', cs['PAY_AMT4'].min(), cs['PAY_AMT4'].max())

input_var = pd.DataFrame()
input_var['AGE'] = [age]
input_var['BILL_AMT1'] = [bill_amt1]
input_var['PAY_AMT6'] = [pay_amt6]
input_var['BILL_AMT6'] = [bill_amt6]
input_var['BILL_AMT5'] = [bill_amt5]
input_var['PAY_AMT1'] = [pay_amt1]
input_var['BILL_AMT2'] = [bill_amt2]
input_var['PAY_AMT4'] = [pay_amt4]


st.markdown("<br>", unsafe_allow_html= True) 
st.subheader('Users Input Variables')
st.divider()
st.dataframe(input_var)

bill_amt1 = joblib.load('BILL_AMT1_scaler.pkl')
pay_amt6 = joblib.load('PAY_AMT6_scaler.pkl')
pay_amt4 = joblib.load('PAY_AMT4_scaler.pkl')
bill_amt5 = joblib.load('BILL_AMT5_scaler.pkl')
pay_amt1 = joblib.load('PAY_AMT1_scaler.pkl')
bill_amt6 = joblib.load('BILL_AMT6_scaler.pkl')

input_var['BILL_AMT6'] = bill_amt6.transform(input_var[['BILL_AMT6']])
input_var['PAY_AMT1'] = pay_amt1.transform(input_var[['PAY_AMT1']])
input_var['BILL_AMT5'] = bill_amt5.transform(input_var[['BILL_AMT5']])
input_var['PAY_AMT4'] = pay_amt4.transform(input_var[['PAY_AMT4']])
input_var['PAY_AMT6'] = pay_amt6.transform(input_var[['PAY_AMT6']])
input_var['BILL_AMT1'] = bill_amt1.transform(input_var[['BILL_AMT1']])

model = joblib.load('CreditCard.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

if st.button('Predict Credit'):
    if predicted == 0:
        st.failure('Customer Has DEFAULTED')
    else:
        st.success('Customer FULLY PAID')
