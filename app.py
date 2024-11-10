import streamlit as st
import numpy as np
import pickle  # Assuming you have the model saved as a pickle file

# Load the pre-trained model (assuming it's saved as 'model.pkl')
with open('gradientboosting_model_pickle.pkl', 'rb') as file:
    gb_model = pickle.load(file)

# Define function to make prediction
def predict_autism(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, age, gender, jaundice, family_asd, relation):
    # Map gender, jaundice, family_asd, and relation to the expected numerical inputs
    gender_m = 1 if gender == 'Male' else 0
    jaundice_yes = 1 if jaundice == 'Yes' else 0
    autism_yes = 1 if family_asd == 'Yes' else 0
    
    # Map relations
    relation_mapping = {
        "Health care professional": [1, 0, 0, 0, 0, 0],
        "Others": [0, 1, 0, 0, 0, 0],
        "Parent": [0, 0, 1, 0, 0, 0],
        "Relative": [0, 0, 0, 1, 0, 0],
        "Self": [0, 0, 0, 0, 1, 0],
        "Family member": [0, 0, 0, 0, 0, 1],
    }
    relation_features = relation_mapping.get(relation, [0, 0, 0, 0, 0, 0])

    # Combine all features into a single array for prediction
    features = np.array([age, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
                         gender_m, 1 - gender_m, jaundice_yes, 1 - jaundice_yes,
                         autism_yes, 1 - autism_yes] + relation_features)
    
    # Predict probability
    probability = gb_model.predict_proba([features])[0][1]
    return probability

# Input fields for the prediction function
q1 = st.number_input('Q1', min_value=0, max_value=1, step=1)
q2 = st.number_input('Q2', min_value=0, max_value=1, step=1)
q3 = st.number_input('Q3', min_value=0, max_value=1, step=1)
q4 = st.number_input('Q4', min_value=0, max_value=1, step=1)
q5 = st.number_input('Q5', min_value=0, max_value=1, step=1)
q6 = st.number_input('Q6', min_value=0, max_value=1, step=1)
q7 = st.number_input('Q7', min_value=0, max_value=1, step=1)
q8 = st.number_input('Q8', min_value=0, max_value=1, step=1)
q9 = st.number_input('Q9', min_value=0, max_value=1, step=1)
q10 = st.number_input('Q10', min_value=0, max_value=1, step=1)
age = st.number_input('Age', min_value=0.0, max_value=100.0, step=0.1)
gender = st.selectbox('Gender', ['Male', 'Female'])
jaundice = st.selectbox('Jaundice', ['Yes', 'No'])
family_asd = st.selectbox('Family history of ASD', ['Yes', 'No'])
relation = st.selectbox('Relation', [
    "Health care professional", "Others", "Parent", "Relative", "Self", "Family member"
])

# Define HTML for output messages
safe_html = """
<div style="background-color: #F4D03F; padding: 10px;">
<h2 style="color: white;text-align:center;"> You do not have ASD</h2>
</div>
"""
danger_html = """
<div style="background-color: #F08080; padding: 10px;">
<h2 style="color: black; text-align:center;"> You may have ASD!! Please consult a professional Doctor.</h2>
</div>
"""

# Run prediction only if user presses button
if st.button("Predict"):
    # Call the prediction function
    output = predict_autism(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, age, gender, jaundice, family_asd, relation)
    st.success('The probability of having ASD is {:.2f}'.format(output))

    # Check output value and display the appropriate message
    if output >= 0.5:
        st.markdown(danger_html, unsafe_allow_html=True)
    else:
        st.markdown(safe_html, unsafe_allow_html=True)
