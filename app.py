from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__, '/static')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')
@app.route('/styles', methods=['GET'])
def styles():
    return render_template('styles.html')

@app.route('/', methods=['POST'])
def calculate():
    imagefile=request.files['imagefile']
    image_path = "./static/images/"+imagefile.filename
    imagefile.save(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    input_image = Image.open(imagefile)

    input_image=transform(input_image)
    input_image = input_image.unsqueeze(0)
    model = torch.load('model.pt',map_location='cpu')
    model.eval()
    with torch.no_grad():
        output = model(input_image)
    styles = ["Art Nouveau", "Baroque", "Expressionism", "Impressionism", "Japanese", "Medieval", "Primitivism"]
    max = output[0][0].item()
    style = styles[0]
    for i in range(7):
        if output[0][i].item()>max:
            max = output[0][i].item()
            style = styles[i]
    return render_template('predict.html', prediction = style, filename=imagefile.filename)

if __name__ == '__main__':
    app.run(port=5600, debug=True)