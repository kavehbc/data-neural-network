# pip install streamlit

# streamlit run <python_file.py>

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def main():
    st.title("Streamlit Plots")
    
    df_titanic = pd.read_csv("../data/titanic.csv")
    st.write(df_titanic)
    
    x = st.sidebar.selectbox("X", options=df_titanic.columns)
    y = st.sidebar.selectbox("Y", options=df_titanic.columns)
    color = st.sidebar.selectbox("Color", options=df_titanic.columns)
    
    btn_plot = st.sidebar.button("Plot")
    if btn_plot:
        fig = px.scatter(df_titanic, x=x, y=y, color=color)
        # fig.show()
        st.plotly_chart(fig, use_container_width=True)

        
if __name__ == "__main__":
    main()
    