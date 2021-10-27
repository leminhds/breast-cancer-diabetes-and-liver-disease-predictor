import streamlit as st
import pickle

import numpy as np


def write():
    """
        write content to this app
        """
    st.title('Diabetes Prediction')

    xgb_pickle = open('cancer_model.pkl', 'rb')
    xgb = pickle.load(xgb_pickle)
    xgb_pickle.close()

    st.subheader('Put in your values below so that our AI can give a prediction '
                 'on whether you might have liver disease')

    st.write('This app uses 6 inputs to predict the probability that you have breast cancer')
    age = st.number_input('Your age',
                          min_value=16, max_value=120, value=16,
                          step=1)
    gender = st.radio('Your gender', ['Male', 'Female'])
    total_bilirubin = st.number_input('Total Bilirubin: Your level of Bilirubin in mg/dL',
                                      min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    alkaline_phosphotase = st.number_input('alkaline phosphotase: ALP in IU/L',
                                           min_value=50, max_value=2500, value=50, step=10)
    alamine_aminotransferase = st.number_input('ALT in IU/L',
                                               min_value=0, max_value=1000, value=0, step=5)

    albumin_and_globulin_ratio = st.number_input('A/G ratio',
                                                 min_value=0.0, max_value=3.5, value=0.0, step=0.1)

    if st.button('Predict'):
        st.write(f'You have selected the values: '
                 f'{[age, gender, total_bilirubin, alkaline_phosphotase, alamine_aminotransferase, albumin_and_globulin_ratio]}')
        gender = 0 if gender == 'Female' else 1
        Xnew = [
            [age, gender, total_bilirubin, alkaline_phosphotase, alamine_aminotransferase, albumin_and_globulin_ratio]]
        # reformat so that it can be feed into xgb predict
        Xnew = np.array(Xnew).reshape((1, -1))
        prediction = xgb.predict_proba(Xnew)

        # what the model predict
        highest_prob = np.argmax(prediction)
        # the actual certainty of prediction
        highest_value = np.amax(prediction)
        if highest_prob == 1:
            predict_value = "have Liver Disease "
        else:
            predict_value = "do not have Liver Disease"
        st.subheader("We predict that you {} with {:.2f}% certainty".format(predict_value, highest_value * 100))
        st.write('Note: The model had an accuracy during training of 86%')


if __name__ == "__main__":
    write()
