import streamlit as st
import pickle
import numpy as np



def write():
    """
    write content to this app
    """
    st.title('Breast Cancer Prediction')

    xgb_pickle = open('cancer_model.pkl', 'rb')
    xgb = pickle.load(xgb_pickle)
    xgb_pickle.close()

    st.subheader('Put in your values below so that our AI can give a prediction '
                 'on whether you might have breast cancer')

    st.write('This app uses 5 inputs to predict the probability that you have breast cancer')
    concave_points_mean = st.number_input('Concave point mean: mean for number of concave portions of the contour',
                                          min_value=0.000, max_value=0.250, value=0.000,
                                          step=0.001, format="%.5f")
    area_mean = st.number_input('Area mean: The mean area of the cells',
                                min_value=0.0, max_value=2500.0, value=0.0, step=1.0, format="%.2f")
    radius_mean = st.number_input('Radius mean: mean of distances from center to points on the perimeter',
                                  min_value=0.0, max_value=40.0, value=0.0, step=0.1, format="%.2f")
    perimeter_mean = st.number_input('Perimeter mean: mean size of the core tumor',
                                     min_value=0.0, max_value=200.0, value=0.0, step=1.0, format="%.2f")
    concavity_mean = st.number_input('Concavity mean: mean of severity of concave portions of the contour',
                                     min_value=0.0, max_value=0.6, value=0.0, step=0.001, format="%.5f")

    if st.button('Predict'):
        st.write(
            f'You have selected the values: {[concave_points_mean, area_mean, radius_mean, perimeter_mean, concavity_mean]}')
        Xnew = [[concave_points_mean, area_mean, radius_mean, perimeter_mean, concavity_mean]]
        # reformat so that it can be feed into xgb predict
        Xnew = np.array(Xnew).reshape((1, -1))
        prediction = xgb.predict_proba(Xnew)

        # what the model predict
        highest_prob = np.argmax(prediction)
        # the actual certainty of prediction
        highest_value = np.amax(prediction)
        if highest_prob == 1:
            predict_value = "have Cancer"
        else:
            predict_value = "do not have Cancer"
        st.subheader("We predict that you {} with {:.3f}% certainty".format(predict_value, highest_value * 100))
        st.write('Note: The model had an accuracy during training of 95%')



if __name__ == "__main__":
    write()
