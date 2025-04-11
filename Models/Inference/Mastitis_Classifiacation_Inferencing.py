import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class MastitisDiseasesClassification:
    def __init__(self, model_path, class_names, input_size=(224, 224), device='cpu'):
        """
        Initialize the Mastitis Diseases Classification class.

        Args:
            model_path (str): Path to the pre-trained model weights.
            class_names (list): List of class names for classification.
            input_size (tuple): Input size for the model.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        self.class_names = class_names
        self.input_size = input_size

        # Load and modify EfficientNet model
        self.model = models.efficientnet_b0()
        self.model.classifier[1] = nn.Linear(
            in_features=self.model.classifier[1].in_features,
            out_features=len(self.class_names)
        )

        # Load the model weights and map to CPU
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def infer(self, image_path):
        """
        Perform inference on an input image.

        Args:
            image_path (str): Path to the input image.
        Returns:
            dict: Predicted label and confidence.
        """
        img = Image.open(image_path).convert('RGB')  # Ensure 3-channel image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)

            predicted_label = self.class_names[preds[0]]
            confidence_score = confidence[0].item() * 100

            return {
                "Predicted Label": predicted_label,
                "Confidence": round(confidence_score, 4)
            }

# ---------------------- USAGE EXAMPLE ----------------------
#
# class_names = ["Normal", "Infected"]
# model_path = r"C:\Users\Abdo\Projects\pythonProject\Models\Inference\Model\Mastitis_Diseases_Classfication_Model.pth"
# image_path = r"C:\Users\Abdo\Projects\pythonProject\Data Samples\Mastitis Disease\Infected Udders\Burnt-teat-e1515283231199-400x284_jpg.rf.9f4b1478dded9d31e2d5e85ca0c8e3f7.jpg"
#
# classifier = MastitisDiseasesClassification(model_path=model_path, class_names=class_names, device='cpu')
# result = classifier.infer(image_path)
#
# print("🔍 Inference Result:")
# print(result)
