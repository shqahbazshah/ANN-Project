import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
model = tf.keras.models.load_model('model.h5')
with open('onehotencoding_geo.pkl', 'rb') as f:
    onehotencoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoding_gender.pkl', 'rb') as f:
    labelencoder_gender = pickle.load(f)

st.title('Customer Churn Prediction')
geography = st.selectbox('Geography', onehotencoder_geo.categories_[0])
gender = st.selectbox('Gender', labelencoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
input_data = {
    'CreditScore': credit_score,
    'Gender': labelencoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
geo_encoded = onehotencoder_geo.transform([[geography]])
import pandas as pd
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=onehotencoder_geo.get_feature_names_out(['Geography']))
input_df = pd.DataFrame([input_data])
final_input = pd.concat([input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)
scaled_input = scaler.transform(final_input)
prediction_proba = model.predict(scaled_input)[0][0]
if prediction_proba > 0.5:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')
st.write(f'Churn probability: {prediction_proba:.2f}')