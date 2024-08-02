import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)

    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return labels[index[0]], percentage[index[0]].item()

if __name__ == "__main__":
    url = "https://your-image-url.com/vehicle.jpg"
    label, confidence = classify_image(url)
    print(f"Label: {label}, Confiance: {confidence:.2f}%")

