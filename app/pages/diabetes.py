import streamlit as st
import pickle

import numpy as np

def write():
    """
        write content to this app
        """
    st.title('Diabetes Prediction')

    xgb_pickle = open('../diabetes_model.pkl', 'rb')
    xgb = pickle.load(xgb_pickle)
    xgb_pickle.close()

    st.subheader('Put in your values below so that our AI can give a prediction '
                 'on whether you might have diabetes')

    st.write('This app uses 8 inputs to predict the probability that you have breast cancer')
    pregnancies = st.number_input('Pregnancies: Number of time pregnant',
                                  min_value=0, max_value=100, value=0,
                                  step=1)
    glucose = st.number_input('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
                              min_value=0, max_value=250, value=0, step=1)
    skinThickness = st.number_input('Skin Thickness: Triceps skin fold thickness (mm)',
                                    min_value=0, max_value=120, value=0, step=1)
    Insulin = st.number_input('Insulin: 2-Hour serum insulin (mu U/ml)',
                              min_value=0, max_value=1000, value=0, step=1)
    BMI = st.number_input('BMI: Body mass index (weight in kg/(height in m)^2)',
                          min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
    diabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction: mean of severity of concave portions of the contour',
                                               min_value=0.0, max_value=3.0, value=0.0, step=0.001, format="%.3f")
    age = st.number_input('BMI: mean of severity of concave portions of the contour',
                          min_value=0, max_value=200, value=18, step=1)

    if st.button('Predict'):
        st.write(
            f'You have selected the values: {[pregnancies, glucose, skinThickness, Insulin, BMI, diabetesPedigreeFunction, age]}')
        Xnew = [[pregnancies, glucose, skinThickness, Insulin, BMI, diabetesPedigreeFunction, age]]
        # reformat so that it can be feed into xgb predict
        Xnew = np.array(Xnew).reshape((1, -1))
        prediction = xgb.predict_proba(Xnew)

        # what the model predict
        highest_prob = np.argmax(prediction)
        # the actual certainty of prediction
        highest_value = np.amax(prediction)
        if highest_prob == 1:
            predict_value = "have Diabetes"
        else:
            predict_value = "do not have Diabetes"
        st.subheader("We predict that you {} with {:.2f}% certainty".format(predict_value, highest_value * 100))
        st.write('Note: The model had an accuracy during training of 91%')


if __name__ == "__main__":
    write()