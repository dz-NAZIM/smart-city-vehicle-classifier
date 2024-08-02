from flask import Flask, request, render_template, redirect
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

def classify_image(img_path):
    img = Image.open(img_path)
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return labels[index[0]], percentage[index[0]].item()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            label, confidence = classify_image(file_path)
            return render_template("index.html", label=label, confidence=confidence, img_path=file_path)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
