import torch
import torchvision.models as models
import torchvision.transforms as T
import kornia.augmentation as K
from PIL import Image
import requests
from io import BytesIO
from kornia.contrib import analyze_model_behavior

model = models.resnet18(weights="DEFAULT").eval()

# Load image from URL
url = "https://picsum.photos/224"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
image = transform(img).unsqueeze(0)

aug = K.RandomAffine(degrees=45.0, translate=(0.3, 0.3))

results = analyze_model_behavior(model, image, augmentation=aug, layers=("layer1", "layer2"))

print("Results:", results)
