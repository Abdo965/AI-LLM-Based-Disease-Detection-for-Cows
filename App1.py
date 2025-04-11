import os
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBqZTixI9qEM0VkCRp3fWXm5EgZ87_ndp8"
import  google.generativeai as geni
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import classifier functions
from Models.Inference.lumpy_skin_classifiacation_inferencing import LumpyDiseasesClassification
from Models.Inference.Mastitis_Classifiacation_Inferencing import MastitisDiseasesClassification

def infer_abdominal_Lumpy(image_path):
    model_path = r"C:\Users\Abdo\Projects\pythonProject\Models\Inference\Model\lumpy_skin.pth"
    Lumpy_detector = LumpyDiseasesClassification(model_path=model_path, class_names = ['Lumpy Skin', 'Normal Skin'])
    result = Lumpy_detector.infer(image_path)
    description = (

        "The system monitors key diseases in cows, including Lumpy Skin Disease, providing farmers with early "
        "disease detection alerts. "
        "The output includes a prediction dictionary indicating the presence (1) or absence (0) of the diseases, along with "
        "confidence scores for each prediction."
    )

    return result, description


def infer_abdominal_Mastitis(image_path):
    model_path = r"C:\Users\Abdo\Projects\pythonProject\Models\Inference\Model\Mastitis_Diseases_Classfication_Model.pth"
    Mastitis_detector = LumpyDiseasesClassification(model_path=model_path,class_names = ["Normal", "Infected"])
    result = Mastitis_detector.infer(image_path)
    description = (

        "The system monitors key diseases in cows, including Mastitis  Disease, providing farmers with early "
        "disease detection alerts. "
        "The output includes a prediction dictionary indicating the presence (1) or absence (0) of the diseases, along with "
        "confidence scores for each prediction."
    )

    return result, description


def generate_radiology_report(classifier_description, classifier_outputs, cow_name, cow_age, cow_breed, date, disease_detected):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    template = """
    You are an expert veterinary report generator specializing in the detection of livestock diseases using advanced IoT-based monitoring. Using the input provided below, create a detailed and professional health report for the cow, focusing on early disease detection and prevention.

    Input details:
    - Classifier Function: {classifier_description}
    - Classifier Results: {classifier_outputs}
    - Cow Name: {cow_name}
    - Cow Age: {cow_age}
    - Cow Breed: {cow_breed}
    - Date: {date}
    - Disease Detected: {disease_detected}

    Expected Report Structure:

    1. Disease Name: {disease_detected}

    2. Brief Description:  
    A simple explanation of the disease based on the classifier results ({classifier_outputs}) and its detection logic ({classifier_description}).

    3. Causes of the Disease:  
    Probable causes as identified by the system or veterinary input based on the sensor data.

    4. Symptoms Observed:  
    Symptoms detected by IoT sensors, such as changes in temperature, movement, behavior, or feeding.

    5. Is it Contagious?  
    State whether the disease can spread to other animals and how.

    6. Severity of the Disease:  
    Explain the seriousness of the disease and possible complications if left untreated.

    7. What Can the Farmer Do? (Initial Advice):  
    Immediate steps the farmer can take, such as isolation, hygiene measures, or dietary changes.

    8. Does it Require Immediate Veterinary Attention?  
    Yes / No — with a short explanation of when and why veterinary intervention is necessary.

    Please generate the report in a clear and professional structure, providing accurate veterinary insights based on smart data analysis."""

    prompt = PromptTemplate(
        input_variables=[
            "classifier_description",
            "classifier_outputs",
            "cow_name",
            "cow_age",
            "cow_breed",
            "date",
            "disease_detected"
        ],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    report = chain.run(
        classifier_description=classifier_description,
        classifier_outputs=classifier_outputs,
        cow_name=cow_name,
        cow_age=cow_age,
        cow_breed=cow_breed,
        date=date,
        disease_detected=disease_detected
    )

    return report



# ......................................... Streamlit app...................................................

# Title for the app
st.title("Cow Health Report Generator")

# Upload image (Simulate uploading the cow's health image for disease classification)
uploaded_image = st.file_uploader("Upload Cow's Health Image", type=["jpg", "jpeg", "png"])

# Classifier options (Simulated function names for disease detection)
classifier_options = {
    "Lumpy Skin Disease": infer_abdominal_Lumpy,
    "Mastitis Disease": infer_abdominal_Mastitis
}

# Select classifier for disease detection
selected_classifier = st.selectbox("Select Classifier", list(classifier_options.keys()))

# Cow information inputs
cow_name = st.text_input("Cow Name")
cow_age = st.number_input("Cow Age (in months)", min_value=0, max_value=200, step=1)
cow_breed = st.text_input("Cow Breed")
date = st.date_input("Examination Date")

# Button to generate health report
if st.button("Generate Health Report"):
    # Check if all necessary inputs are provided
    if uploaded_image and cow_name and cow_age and cow_breed:
        # Save uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.read())

        # Get the selected classifier's function to process the image
        classifier_function = classifier_options[selected_classifier]
        classifier_description, classifier_outputs = classifier_function("temp_image.jpg")

        # Assign the disease detected based on the classifier selected
        disease_detected = selected_classifier

        # Generate the health report using the classifier's output
        report = generate_radiology_report(
            classifier_description=classifier_description,
            classifier_outputs=classifier_outputs,
            cow_name=cow_name,
            cow_age=cow_age,
            cow_breed=cow_breed,
            date=date,
            disease_detected=disease_detected
        )

        # Display the generated health report
        st.subheader("Generated Health Report for Cow")
        st.markdown(report)
    else:
        # If any field is missing or image is not uploaded, show an error
        st.error("Please complete all fields and upload an image.")
