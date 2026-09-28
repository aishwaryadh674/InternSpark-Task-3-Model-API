from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io

app = FastAPI(
    title="CIFAR-10 Image Classification API",
    description="API for CIFAR-10 image classification using a trained ResNet18 model",
    version="1.0"
)

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Create ResNet18 architecture
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# Load trained model
model.load_state_dict(
    torch.load("cifar10_resnet18.pth", map_location="cpu")
)

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/")
def home():
    return {
        "message": "CIFAR-10 Image Classification API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_class_id = torch.argmax(
            probabilities, dim=1
        ).item()

        confidence = probabilities[0][predicted_class_id].item()

    return {
        "filename": file.filename,
        "predicted_class_id": predicted_class_id,
        "predicted_class": class_names[predicted_class_id],
        "confidence": round(confidence, 4)
    }