import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


st.set_option('deprecation.showPyplotGlobalUse', False)
model = pickle.load(open(os.path.join(os.getcwd(), 'model', 'Bias_MLP_model.pkl'), 'rb'))
# output = [0, 0, 2, 3, 4, 3, 1, 5, 3, 5, 5, 4, 3, 4, 4, 5]

# test = np.array([output])

# model.predict(test)


def predict_MLP(evaluation: int) -> float:

    input = np.array(evaluation)
    prediction = model.predict(input)
    # ponto = (prediction - model.intercept_)/model.coef_[0]

    return prediction


def main():

    clf_data = list()
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    Multi-Layer Perceptron App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    # customer_type = st.text_input("SELECT TYPE")
    # st.markdown('**SELECT THE CUSTOMER TYPE**')

    gender = st.radio('1. SELECT GENDER',
        (
        '1. Male',
        '2. Female')
        )

    customer_type = st.radio('2. SELECT THE CUSTOMER TYPE',
        (
        '1. Loyal Customer',
        '2. Disloyal Customer')
        )
    
    age = st.text_input('3. Type your age')

    type_of_travel = st.radio('4. SELECT THE TYPE OF TRAVEL',
        (
         '1. Personal',
         '2. Business')
         )
    
    customer_class = st.radio('5. SELECT THE CUSTOMER CLASS',
        (
        '1. Business',
        '2. Eco',
        '3. Eco Plus')
        )

    flight_distance = st.text_input('6 . Type your flight distance')

    inflight_wifi_service = st.radio('7. EVALUATE INFLIGHT WIFI SERVICE',
        (
            1,
            2,
            3,
            4,
            5
        )
        )

    departure_arrival_time_convenient = st.radio('8. EVALUATE TIME CONVENIENT (DEPARTURE/ARRIVAL)',
        (
            1,
            2,
            3,
            4,
            5
        )
        )
    
    ease_of_online_booking = st.radio('9. EVALUATE EASE OF ONLINE BOOKING',
        (
            1,
            2,
            3,
            4,
            5
        )
        )
    
    gate_location = st.radio('10. EVALUATE THE GATE LOCATION',
        (
            1,
            2,
            3,
            4,
            5
        )
        )

    food_and_drink = st.radio('11. EVALUATE THE FOOD AND DRINK',
        (
            1,
            2,
            3,
            4,
            5
        )
        )

    online_boarding = st.radio('12. EVALUATE THE ONLINE BOARDING',
        (
            1,
            2,
            3,
            4,
            5
        )
        )

    seat_comfort = st.radio('13. EVALUATE THE SEAT COMFORT',
        (
            1,
            2,
            3,
            4,
            5
        )
        )
    
    inflight_entertainment = st.radio('14. EVALUATE THE INFLIGHT ENTERTAIMENT',
        (
            1,
            2,
            3,
            4,
            5
        )
        )
    
    on_board_service = st.radio('15. EVALUATE THE ONBOARD SERVICE',
        (
            1,
            2,
            3,
            4,
            5
        )
        )
    
    leg_room_service = st.radio('16. EVALUATE THE LEG ROOM SERVICE',
        (
            1,
            2,
            3,
            4,
            5
        )
        )

    baggage_handling = st.radio('17. EVALUATE THE BAGGAGE HANDLING',
        (
            1,
            2,
            3,
            4,
            5
        )
        )
    
    checkin_service = st.radio('18. EVALUATE THE CHECK-IN SERVICE',
        (
            1,
            2,
            3,
            4,
            5
        )
        )
    
    inflight_service = st.radio('19. EVALUATE THE INFLIGHT SERVICE',
        (
            1,
            2,
            3,
            4,
            5
        )
        )
    
    cleanliness = st.radio('20. EVALUATE THE CLEANLINESS',
        (
            1,
            2,
            3,
            4,
            5
        )
        )

    departure_delay_in_minutes = st.text_input('21 . Type your delay in minutes')

    if gender == 'Male':
        gender = 1
    else:
        gender = 0

    if customer_type == 'Loyal Customer':
        customer_type = 0
    else:
        customer_type = 1
    
    if customer_class == 'Business':
        customer_class = 0
    elif customer_class == 'Eco':
        customer_class = 1
    else:
        customer_class = 2
    
    if type_of_travel == 'Personal':
        type_of_travel = 1 
    else:
        type_of_travel = 0

    clf_data.append([gender, customer_type, int(age), type_of_travel,
                    customer_class, int(flight_distance), inflight_wifi_service, departure_arrival_time_convenient,
                    ease_of_online_booking, gate_location, food_and_drink, online_boarding,
                    seat_comfort, inflight_entertainment, on_board_service, leg_room_service, baggage_handling,
                    checkin_service, inflight_service, cleanliness, departure_delay_in_minutes])
   
    if st.button('Classifier'):
        prediction = predict_MLP(clf_data)
        if prediction == 1:
            st.success(f'The customer is "satisfied"')
        else:
            st.success(f'The customer is "dissatisfied"')

if __name__ == '__main__':
    main()
    