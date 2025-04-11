import os
import google.generativeai as geni
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import classifier functions
from Models.Inference.lumpy_skin_classifiacation_inferencing import LumpyDiseasesClassification
from Models.Inference.Mastitis_Classifiacation_Inferencing import MastitisDiseasesClassification

# Set Google API Key securely (outside code in a real-world scenario)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBqZTixI9qEM0VkCRp3fWXm5EgZ87_ndp8"

def infer_abdominal_Lumpy(image_path):
    model_path = r"C:\Users\Abdo\Projects\pythonProject\Models\Inference\Model\lumpy_skin.pth"
    Lumpy_detector = LumpyDiseasesClassification(model_path=model_path)
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
    Mastitis_detector = MastitisDiseasesClassification(model_path=model_path)
    result = Mastitis_detector.infer(image_path)
    description = (
        "The system monitors key diseases in cows, including Mastitis Disease, providing farmers with early "
        "disease detection alerts. "
        "The output includes a prediction dictionary indicating the presence (1) or absence (0) of the diseases, along with "
        "confidence scores for each prediction."
    )
    return result, description

def generate_radiology_report(classifier_description, classifier_outputs, cow_name, cow_age, cow_breed, date, disease_detected):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")  # Check API setup

    # Template updated for IoT-based livestock monitoring and disease detection
    template = """
    You are an expert veterinary report generator specializing in the detection of livestock diseases using advanced IoT-based monitoring. 
    Using the input provided below, create a detailed and professional health report for the cow, focusing on early disease detection and prevention:

    1. A description of the classifier results.
    2. Relevant veterinary recommendations based on the classifier outputs.
    3. If a disease is detected, include a note about potential complications or outcomes if left untreated.

    Input details:
    - Classifier Function: {classifier_description}
    - Classifier Results: {classifier_outputs}
    - Cow Name: {cow_name}
    - Cow Age: {cow_age}
    - Cow Breed: {cow_breed}
    - Date: {date}
    - Disease Detected: {disease_detected}

    Generate the report with a clear structure and provide the necessary veterinary insights.

    Expected Report Structure:
    - **Cow Information**: Include the cow's name, age, breed, and date.
    - **Findings**: Summarize the classifier results.
    - **Impression**: Provide a concise interpretation of the findings.
    - **Recommendations**: Suggest next steps or treatments.
    - **Warnings**: If a disease is detected, highlight potential outcomes if untreated. If no disease is detected, provide preventive advice to ensure the cow's health.
    """

    prompt = PromptTemplate(
        input_variables=["classifier_description", "classifier_outputs", "cow_name", "cow_age", "cow_breed", "date", "disease_detected"],
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

# ......................................... Streamlit app ...................................................

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


# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import transforms
# import torch.nn as nn
# import torchvision.models as models
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.llms import ChatGoogleGenerativeAI
#
# # Define LumpyDiseasesClassification class
# class LumpyDiseasesClassification:
#     def __init__(self, model_path, class_names, input_size=(224, 224), device=None):
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.class_names = class_names
#         self.input_size = input_size
#
#         # Initialize and modify the EfficientNet model
#         self.model = models.efficientnet_b0()
#         self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=len(self.class_names))
#         state_dict = torch.load(model_path, map_location=self.device)
#         new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#         self.model.load_state_dict(new_state_dict)
#         self.model.to(self.device)
#         self.model.eval()
#
#         # Define image transformations
#         self.transform = transforms.Compose([
#             transforms.Resize(self.input_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#     def infer(self, image_path):
#         img = Image.open(image_path).convert('RGB')
#         img_tensor = self.transform(img).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             outputs = self.model(img_tensor)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             confidence, pred_class = torch.max(probs, 1)
#             predicted_label = self.class_names[pred_class.item()]
#             confidence_score = confidence.item() * 100
#             return {'class': predicted_label, 'probability': round(confidence_score, 4)}
#
# # Define MastitisDiseasesClassification class
# class MastitisDiseasesClassification:
#     def __init__(self, model_path, class_names, input_size=(224, 224), device='cpu'):
#         self.device = torch.device(device)
#         self.class_names = class_names
#         self.input_size = input_size
#
#         # Load and modify EfficientNet model
#         self.model = models.efficientnet_b0()
#         self.model.classifier[1] = nn.Linear(in_features=self.model.classifier[1].in_features, out_features=len(self.class_names))
#         state_dict = torch.load(model_path, map_location=self.device)
#         new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#         self.model.load_state_dict(new_state_dict)
#         self.model.to(self.device)
#         self.model.eval()
#
#         # Define transformations
#         self.transform = transforms.Compose([
#             transforms.Resize(self.input_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#     def infer(self, image_path):
#         img = Image.open(image_path).convert('RGB')
#         img_tensor = self.transform(img).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             outputs = self.model(img_tensor)
#             probabilities = torch.softmax(outputs, dim=1)
#             confidence, preds = torch.max(probabilities, 1)
#             predicted_label = self.class_names[preds[0]]
#             confidence_score = confidence[0].item() * 100
#             return {"Predicted Label": predicted_label, "Confidence": round(confidence_score, 4)}
#
# # LangChain to generate the radiology report
# def generate_radiology_report(classifier_description, classifier_outputs, cow_name, cow_age, cow_breed, date, disease_detected):
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
#
#     template = """
#     You are a veterinary radiologist interpreting X-ray results of a cow using an AI-powered classifier. Based on the classifier outputs and cow information, you will generate a comprehensive radiology report that includes the following:
#
#     1. **Classifier Results Interpretation**: Summarize the classifier’s outputs and provide a detailed description of the findings.
#     2. **Disease Detection**: If a disease is detected, describe the condition, potential complications, and possible outcomes if untreated.
#     3. **Veterinary Recommendations**: Provide relevant veterinary recommendations or treatments based on the classifier’s results.
#     4. **Warnings**: If no disease is detected, emphasize preventive measures or the need for further observation.
#
#     Input details:
#     - Classifier Function: {classifier_description}
#     - Classifier Results: {classifier_outputs}
#     - Cow Name: {cow_name}
#     - Cow Age: {cow_age}
#     - Cow Breed: {cow_breed}
#     - Date: {date}
#     - Disease Detected: {disease_detected}
#
#     Generate a structured and professional veterinary radiology report with the following sections:
#     - **Cow Information**: Include the cow's name, age, breed, and the date of the report.
#     - **Findings**: Summarize the classifier’s results and provide an interpretation based on the classifier outputs.
#     - **Impression**: Provide a brief but precise interpretation of the findings.
#     - **Recommendations**: Suggest potential veterinary treatments or actions to take.
#     - **Warnings**: If a disease is detected, list potential outcomes if untreated. If no disease is detected, provide preventive advice to ensure the cow's health.
#     """
#
#     prompt = PromptTemplate(
#         input_variables=["classifier_description", "classifier_outputs", "cow_name", "cow_age", "cow_breed", "date", "disease_detected"],
#         template=template
#     )
#
#     chain = LLMChain(llm=llm, prompt=prompt)
#     report = chain.run(
#         classifier_description=classifier_description,
#         classifier_outputs=classifier_outputs,
#         cow_name=cow_name,
#         cow_age=cow_age,
#         cow_breed=cow_breed,
#         date=date,
#         disease_detected=disease_detected
#     )
#
#     return report
#
# # Streamlit app interface
# st.title('Veterinary Disease Classification & Report Generation')
#
# st.sidebar.header('Upload Image')
# uploaded_image = st.sidebar.file_uploader("Choose an image...", type="jpg")
#
# if uploaded_image is not None:
#     st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
#
#     # Select the disease type for classification
#     disease_type = st.radio("Select Disease Type", ('Lumpy Skin', 'Mastitis'))
#
#     if disease_type == 'Lumpy Skin':
#         classifier = LumpyDiseasesClassification(model_path="path/to/lumpy_skin_model.pth", class_names=['Lumpy Skin', 'Normal'])
#     else:
#         classifier = MastitisDiseasesClassification(model_path="path/to/mastitis_model.pth", class_names=['Normal', 'Infected'])
#
#     result = classifier.infer(uploaded_image)
#     st.write(f"Prediction: {result['class']} with confidence: {result['probability']}%")
#
#     # Generate the radiology report
#     if result['class'] != 'Normal':
#         disease_detected = "Yes"
#     else:
#         disease_detected = "No"
#
#     cow_name = st.text_input("Cow's Name")
#     cow_age = st.text_input("Cow's Age")
#     cow_breed = st.text_input("Cow's Breed")
#     date = st.text_input("Date of Report")
#
#     if st.button('Generate Report'):
#         report = generate_radiology_report(
#             classifier_description="Disease classification model",
#             classifier_outputs=str(result),
#             cow_name=cow_name,
#             cow_age=cow_age,
#             cow_breed=cow_breed,
#             date=date,
#             disease_detected=disease_detected
#         )
#
#         st.subheader('Generated Radiology Report')
#         st.write(report)
#
#
# # import os
# # if "GOOGLE_API_KEY" not in os.environ:
# #     os.environ["GOOGLE_API_KEY"] = ""
# #
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.prompts import PromptTemplate
# # from langchain.chains import LLMChain
# #
# # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
# # template = "explain the topic {topic}"
# #
# #
# # prompt = PromptTemplate(
# #         input_variables=["topic"],
# #         template=template )
# #
# # chain = LLMChain(llm=llm, prompt=prompt)
# #
# # output=chain.run({"topic":"Artificai intelligence"})
# # print(output)