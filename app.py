
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
# Ensure 'happiness_model.pkl' is in the same directory as this app.py when deploying
model = joblib.load('happiness_model.pkl')

st.title('Mental Health and Social Media Balance Predictor')
st.write('Predict Happiness Index, Sleep Quality, and Stress Level based on user input.')

# User input fields
age = st.slider('Age', 15, 80, 25)

gender_options = ['Female', 'Male', 'Other'] # Based on your dataset's possible values
gender_selected = st.selectbox('Gender', gender_options)

screen_time = st.slider('Daily Screen Time (hours)', 0.0, 10.0, 3.0, 0.1)
days_without_social = st.slider('Days Without Social Media', 0.0, 7.0, 2.0, 0.1)
exercise_freq = st.slider('Exercise Frequency (per week)', 0.0, 7.0, 3.0, 0.1)

platform_options = ['Facebook', 'YouTube', 'TikTok', 'Instagram', 'LinkedIn', 'X (Twitter)'] # Based on your dataset
platform_selected = st.selectbox('Social Media Platform', platform_options)

# Encode categorical inputs
# Recreate LabelEncoders or use predefined mappings
# For Gender: 'Female' will be 0, 'Male' 1, 'Other' 2 IF your original df['Gender'] had those in that order
# Based on typical alphabetical order for fit_transform: Female (0), Male (1), Other (2)
# However, it depends on the actual unique values in your original 'df["Gender"]' column
# Let's assume the order from original fit_transform for consistency:
# Original encoding: Male, Female, Other -- let's reconfirm the order from notebook's `le.classes_`
# In the notebook, `df["Gender"]` was encoded. If your original data had 'Male' and 'Female' and 'Other',
# `le.fit_transform` would encode them alphabetically. Let's assume Male=1, Female=0, Other=2 (if they exist).

# To be robust, you should save the LabelEncoder objects or their classes mapping.
# For this example, let's create simple mappings based on common practice or observed notebook behavior:

gender_mapping = {
    'Female': 0,
    'Male': 1,
    'Other': 2
}

platform_mapping = {
    'Facebook': 0,
    'Instagram': 1,
    'LinkedIn': 2,
    'TikTok': 3,
    'X (Twitter)': 4,
    'YouTube': 5
}

gender_encoded = gender_mapping.get(gender_selected)
platform_encoded = platform_mapping.get(platform_selected)


if st.button('Predict'):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender_encoded,
        'Daily_Screen_Time(hrs)': screen_time,
        'Days_Without_Social_Media': days_without_social,
        'Exercise_Frequency(week)': exercise_freq,
        'Social_Media_Platform': platform_encoded
    }])

    # Make prediction
    prediction = model.predict(input_data)

    st.subheader('Prediction Results:')
    st.write(f"Happiness Index (1-10): {prediction[0][0]:.2f}")
    st.write(f"Sleep Quality (1-10): {prediction[0][1]:.2f}")
    st.write(f"Stress Level (1-10): {prediction[0][2]:.2f}")
