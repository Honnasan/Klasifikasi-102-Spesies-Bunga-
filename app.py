from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import os
import json
import io
import base64

app = Flask(__name__)

# Load class names from JSON
CLASS_JSON_PATH = os.path.join('model', 'cat_to_name.json')
with open(CLASS_JSON_PATH, 'r') as f:
    cat_to_name = json.load(f)
# Urutkan berdasarkan key (asumsi key adalah string angka index kelas)
CLASS_NAMES = [cat_to_name[k] for k in sorted(cat_to_name, key=lambda x: int(x))]
NUM_CLASSES = len(CLASS_NAMES)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_model():
    model = models.mobilenet_v3_large(pretrained=False)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join('model', 'mobilenetv3_flower_best.pth'), map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert('RGB')
    return transform(image).unsqueeze(0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Proses prediksi langsung dari file upload (tanpa simpan ke disk)
            img_bytes = file.read()
            input_tensor = transform_image(io.BytesIO(img_bytes))
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, predicted = torch.max(probs, 1)
                label = CLASS_NAMES[predicted.item()]
                confidence = conf.item() * 100
            # Encode gambar ke base64 untuk preview
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            image_url = f"data:image/png;base64,{img_b64}"
            result = f'Predicted: {label} ({confidence:.1f}%)'
            return render_template('result.html', result=result, image_url=image_url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 