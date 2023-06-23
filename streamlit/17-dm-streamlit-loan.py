import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title("Loan Approval")
    
    # loading objects
    model = pickle.load(open("../object/model.pkl", "rb"))
    encoder = pickle.load(open("../object/encoder.pkl", "rb"))
    scaler = pickle.load(open("../object/scaler.pkl", "rb"))
    
    # Getting user inputs
    ApplicantIncome = st.sidebar.number_input("Applicant Income")
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income")
    LoanAmount = st.sidebar.number_input("Loan Amount")
    Loan_Amount_Term = st.sidebar.selectbox("Loan Term", options=[360.0, 120.0, 240.0, 180.0, 60.0, 300.0, 480.0, 36.0, 84.0, 12.0])
    Credit_History = st.sidebar.selectbox("Credit History", options=[1, 0])
    Gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    Married = st.sidebar.checkbox("Married", value="Yes")
    if Married:
        Married = "Yes"
    else:
        Married = "No"
    
    Dependents = st.sidebar.selectbox("Dependents", options=['0', '1', '2', '3+'])
    Education = st.sidebar.selectbox("Education", options=["Graduate", "Not Graduate"])
    Self_Employed = st.sidebar.checkbox("Self Employed", value="Yes")
    
    if Self_Employed:
        Self_Employed = "Yes"
    else:
        Self_Employed = "No"
        
    Property_Area = st.sidebar.selectbox("Property Area", options=['Urban', 'Rural', 'Semiurban'])

    # creating a dataframe containing user inputs
    df = pd.DataFrame(data=[[ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
                                 Credit_History, Gender, Married, Dependents,
                                 Education, Self_Employed, Property_Area]],
                            columns=["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
                                    "Credit_History", "Gender", "Married", "Dependents",
                                    "Education", "Self_Employed", "Property_Area"])
    st.subheader("Original Dataframe")
    st.write(df)
    
    # apply one-hot-encoder
    categorical_features = ["Gender", "Married", "Dependents",
                            "Education", "Self_Employed", "Property_Area"]
    column_names = encoder.get_feature_names_out(categorical_features)
    transformed_data = encoder.transform(df[categorical_features])
    df_transformed = pd.DataFrame.sparse.from_spmatrix(transformed_data,
                                                       columns=column_names)
    df = df.drop(columns=categorical_features)
    df = df.join(df_transformed)
    df[column_names] = df[column_names].sparse.to_dense()
    
    st.subheader("Encoded Dataframe")
    st.write(df)
    
    # apply scaler
    columns_to_scale = ["ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Loan_Amount_Term"]
    
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    
    st.subheader("Scaled Dataframe")
    st.write(df)
    st.write(df.shape)
    
    # apply neural network
    result = model.predict(df)
    
    st.subheader("Result")
    st.write(result)
    
        
if __name__ == "__main__":
    main()
    