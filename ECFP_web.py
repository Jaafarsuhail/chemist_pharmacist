import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
import numpy as np

# Page title and layout configuration
st.set_page_config(
    page_title="Bioactivity Prediction App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Logo image
image = Image.open('logo.png')

# Increase the resolution of the logo
image = image.resize((3000, 3000))  # Adjust the dimensions as desired


# Display the logo image
st.image(image, use_column_width=True)
    
from rdkit import Chem
from rdkit.Chem import AllChem

# Molecular descriptor calculator
def desc_calc(smiles):
    # Convert SMILES to RDKit molecule object
    molecule = Chem.MolFromSmiles(smiles)

    # Generate ECFP fingerprints
    ecfp = AllChem.GetMorganFingerprintAsBitVect(molecule, 3, nBits=2000)

    # Convert fingerprint to numpy array
    desc_ecfp = np.zeros((1,))
    AllChem.DataStructs.ConvertToNumpyArray(ecfp, desc_ecfp)

    # Create dataframe from the descriptor
    desc = pd.DataFrame(desc_ecfp.reshape(1, -1))

# Save the descriptors to a CSV file
    desc.to_csv('descriptors_output.csv', index=False)
    
    return desc

def build_model(descriptors):
    # Load the trained model
    model = pickle.load(open('ECFP_full.pkl', 'rb'))

    # Check if 'AUTOGEN_input' column exists in descriptors
    if 'AUTOGEN_input' in descriptors.columns:
        descriptors = descriptors.drop('AUTOGEN_input', axis=1)

    # Drop any rows with NaN values
    descriptors = descriptors.dropna()

    # Check if any samples are remaining
    if descriptors.shape[0] == 0:
        raise ValueError("No valid samples remaining after preprocessing.")

    # Convert the descriptors to float type, ignoring any conversion errors
    descriptors = descriptors.apply(pd.to_numeric, errors='coerce')

    # Remove any remaining rows with NaN values after conversion
    descriptors = descriptors.dropna()

    # Check if any samples are remaining
    if descriptors.shape[0] == 0:
        raise ValueError("No valid samples remaining after preprocessing.")

    # Convert the descriptors dataframe to numpy array
    descriptors = descriptors.astype(np.float32)

    # Make predictions using the loaded model
    predictions = model.predict(descriptors)

    # Save predictions to a CSV file
    pred_df = pd.DataFrame(predictions, columns=["Prediction"])
    pred_df.to_csv('predictions.csv', index=False)

    return predictions

# Sidebar
with st.sidebar.header('Enter SMILE string'):
    smiles_input = st.sidebar.text_input("SMILE string") 
    # Predict button and calculation
    if st.sidebar.button('Predict'):
        # Calculate descriptors
        desc = desc_calc(smiles_input)
        
        # Build and display the model predictions
        pred = build_model(desc)
        pred_df = pd.DataFrame(pred, columns=["Prediction"])
        st.dataframe(pred_df.style.highlight_max(axis=0))      
    
def create_download_link(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    # Replace the commas with a comma followed by a space for better readability
    data = data.replace(',', ', ')
    encoded_data = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:text/plain;base64,{encoded_data}" download="{os.path.basename(file_path)}.txt">Download {os.path.basename(file_path)}.txt</a>'
    return href

# Page title and description
st.markdown("""
# Bioactivity Prediction App (MCF7 cell line with estrogen receptor alpha overexpression)

Using this app, you can estimate how effective your compound will be at preventing the α-ER from functioning, which will prevent the MCF-7 cell line from dividing. α-ER is overexpressed in the breast cancer cell line MCF-7.

**Contact Details**
For any inquiries or feedback, please contact [Me](mailto:pharmacistjaafar@gmail.com).

**Credits**
- App built in `Python` + `Streamlit` by [Jaafar Suhail](https://scholar.google.com/citations?user=uN-R_YYAAAAJ&hl=en)
---
""", unsafe_allow_html=True)

# Read in calculated descriptors and display the dataframe
st.header('**Calculated molecular descriptors**')
desc = pd.read_csv('descriptors_output.csv')
st.dataframe(desc)  
st.write(desc.shape)

# Apply trained model to make prediction on query compounds
predictions = build_model(desc)

# Display the predictions
st.header('**Model Predictions**')
predictions_df = pd.DataFrame(predictions, columns=["Prediction"])
st.dataframe(predictions_df.style.highlight_max(axis=0))

# Provide a download link for the predictions
st.markdown(create_download_link('predictions.csv'), unsafe_allow_html=True)
