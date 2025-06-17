import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request
import torchvision.models as models
from werkzeug.utils import secure_filename
import os
import torch.nn.functional as F
from bitirme_web.db import SessionLocal, BreakhisPrediction, BrainPrediction

app = Flask(__name__)

# --- Cihaz Tanımı (GPU varsa GPU, yoksa CPU kullan) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- MODEL DOSYA YOLLARI ---
BREAKHIS_MODEL_PATH = 'vgg19.pth'
BRAIN_MODEL_PATH = 'resnet18_brain_model.pth'  # Burayı da senin istediğin model dosyasına çevirdim

# --- SINIFLAR ---
breakhis_classes = ['Benign', 'Malignant']
brain_classes = ['No Tumor', 'Tumor']

# --- DÖNÜŞÜMLER ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- BREAKHIS MODEL ---
vgg19_breakhis = models.vgg19()
vgg19_breakhis.classifier[6] = torch.nn.Linear(4096, 2)
vgg19_breakhis.load_state_dict(torch.load(BREAKHIS_MODEL_PATH, map_location=device))
vgg19_breakhis.to(device)
vgg19_breakhis.eval()

# --- BRAIN MODEL (ResNet18 olarak değiştirdik) ---
resnet18_brain = models.resnet18()
resnet18_brain.fc = torch.nn.Linear(resnet18_brain.fc.in_features, 2)
resnet18_brain.load_state_dict(torch.load(BRAIN_MODEL_PATH, map_location=device))
resnet18_brain.to(device)
resnet18_brain.eval()


@app.route('/')
def index():
    session = SessionLocal()
    try:
        breakhis_records = session.query(BreakhisPrediction).all()
        brain_records = session.query(BrainPrediction).all()
        return render_template("index.html", breakhis_predictions=breakhis_records, brain_predictions=brain_records)
    finally:
        session.close()


@app.route('/breakhis')
def breakhis():
    return render_template('breakhis.html')


@app.route('/brain')
def brain():
    return render_template('brain.html')


@app.route('/predict_breakhis', methods=['POST'])
def predict_breakhis():
    if 'image' not in request.files:
        return "Dosya bulunamadı.", 400

    file = request.files['image']
    if file.filename == '':
        return "Geçerli bir dosya seçilmedi.", 400

    name = request.form.get("name", "")
    surname = request.form.get("surname", "")

    try:
        filename = secure_filename(file.filename)
        os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
        image_path = os.path.join('static', 'uploads', filename)
        file.save(image_path)

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = vgg19_breakhis(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
            prediction = breakhis_classes[predicted.item()]
            percentage = round(confidence.item() * 100, 2)

        # --- Veritabanına kayıt ---
        session = SessionLocal()
        new_record = BreakhisPrediction(
            name=name,
            surname=surname,
            image_path=image_path,
            prediction=prediction,
            confidence=percentage
        )
        session.add(new_record)
        session.commit()
        session.close()

        return render_template('resultBreakhis.html',
                               name=name,
                               surname=surname,
                               prediction=prediction,
                               percentage=percentage,
                               image_url='/' + image_path)

    except Exception as e:
        return f"Hata oluştu: {str(e)}", 500


@app.route('/predict_brain', methods=['POST'])
def predict_brain():
    if 'image' not in request.files:
        return "Dosya bulunamadı.", 400

    file = request.files['image']
    if file.filename == '':
        return "Geçerli bir dosya seçilmedi.", 400

    name = request.form.get("name", "")
    surname = request.form.get("surname", "")

    try:
        filename = secure_filename(file.filename)
        os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
        image_path = os.path.join('static', 'uploads', filename)
        file.save(image_path)

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = resnet18_brain(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
            prediction = brain_classes[predicted.item()]
            percentage = round(confidence.item() * 100, 2)

        # --- Veritabanına kayıt ---
        session = SessionLocal()
        new_record = BrainPrediction(
            name=name,
            surname=surname,
            image_path=image_path,
            prediction=prediction,
            confidence=percentage
        )
        session.add(new_record)
        session.commit()
        session.close()

        return render_template('resultBrain.html',
                               name=name,
                               surname=surname,
                               prediction=prediction,
                               percentage=percentage,
                               image_url='/' + image_path)

    except Exception as e:
        return f"Hata oluştu: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
