import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class LumpyDiseasesClassification:
    def __init__(self, model_path, class_names, input_size=(224, 224), device=None):
        """
        Initialize the Lumpy skin Diseases Classification class.

        Args:
            model_path (str): Path to the pre-trained model weights.
            class_names (list): List of class names for classification.
            input_size (tuple): Input size for the model.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.input_size = input_size

        # Initialize and modify the EfficientNet model
        self.model = models.efficientnet_b0()
        self.model.classifier[1] = nn.Linear(
            in_features=self.model.classifier[1].in_features,
            out_features=len(self.class_names)
        )

        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

        # Move the model to the selected device
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def imshow_fixed_grid1(self, img, labels):
        """
        Displays an image with fixed grid size.

        Args:
            img (torch.Tensor): The input image tensor.
            labels (str): The label corresponding to the input image.
        """
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        npimg = npimg * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        npimg = np.clip(npimg, 0, 1)

        plt.imshow(npimg)
        plt.title(labels)
        plt.axis('off')
        plt.show()

    def infer(self, image_path):
        """
        Perform inference on an input image and return prediction.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: Predicted class and probability.
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)

            predicted_label = self.class_names[pred_class.item()]
            confidence_score = confidence.item()*100

            return {
                'class': predicted_label,
                'probability': round(confidence_score, 4)
            }

#
# # Define class names
# class_names = ['Lumpy Skin', 'Normal Skin']
#
# # Path to model weights
# model_path = r"C:\Users\Abdo\Projects\pythonProject\Models\Inference\Model\lumpy_skin.pth"
#
# # Initialize the detection class
# classifier = LumpyDiseasesClassification(model_path=model_path, class_names=class_names)
#
# # Perform inference
# image_path = r"C:\Users\Abdo\Projects\pythonProject\Data Samples\Lumpy skin disease\Normal skin\imgs001.jpg"
# result = classifier.infer(image_path)
# print(result)
