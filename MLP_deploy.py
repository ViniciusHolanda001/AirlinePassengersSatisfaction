import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


st.set_option('deprecation.showPyplotGlobalUse', False)
model = pickle.load(open(os.path.join(os.getcwd(), 'model', 'mlp_model.pkl'), 'rb'))


def predict_RL(HeadSize: int) -> float:

    input = np.array([[HeadSize]]).astype(np.float64)
    prediction = model.predict(input)
    ponto = (prediction - model.intercept_)/model.coef_[0]

    return float(prediction), ponto


def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    Multi-Layer Perceptron App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # customer_type = st.text_input("SELECT TYPE")
    st.markdown('\n**SELECT THE CUSTOMER TYPE**')
    customer_type = st.radio('',
        (
        '1. Loyal Customer',
        '2. Disloyal Customer')
        )
    
    st.markdown('**SELECT THE CUSTOMER CLASS**')
    customer_type = st.radio('',
        (
        '1. Business',
        '2. Eco',
        '3. Eco Plus')
        )



    cls_button = st.button('Classifier')



if __name__ == '__main__':
    main()
    