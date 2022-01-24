import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)


model = pickle.load(
    open(os.path.join(os.getcwd(), 'model', 'Pickle_MLP_Model.pkl'), 'rb'))


def predict_RL(HeadSize: int) -> float:

    input = np.array([[HeadSize]]).astype(np.float64)


def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    Multi-Layer Perceptron App </h2>
    </div>
    """



if __name__ == '__main__':
    main()