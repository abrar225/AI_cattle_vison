from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
import os
import base64
import numpy as np

app = Flask(__name__)


class AdvancedViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedViTClassifier, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vit(x)


class CattleClassifier:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.breed_names = self.load_breed_names()
        self.model_loaded = False
        self.load_model(model_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_breed_names(self):
        return [
            'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
            'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar',
            'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam',
            'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari',
            'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori',
            'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
            'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda',
            'Umblachery', 'Vechur'
        ]

    def load_model(self, model_path):
        actual_path = '../models/trained/best_model.pth'
        if os.path.exists(actual_path):
            try:
                checkpoint = torch.load(actual_path, map_location=self.device)
                self.model = AdvancedViTClassifier(num_classes=len(self.breed_names))

                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print(f"❌ Model file not found at: {actual_path}")

    def predict(self, image):
        """Predict cattle breed from image - THIS WAS MISSING!"""
        if not self.model_loaded:
            print("❌ Model not loaded")
            return None

        try:
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            # Get top 2 predictions only
            top2_conf, top2_idx = torch.topk(probabilities, 2)

            results = []
            for i in range(2):
                breed_idx = top2_idx[0][i].item()
                if breed_idx < len(self.breed_names):
                    breed_name = self.breed_names[breed_idx]
                else:
                    breed_name = f"Class_{breed_idx}"

                confidence_val = top2_conf[0][i].item()
                results.append({
                    'rank': i + 1,
                    'breed': breed_name,
                    'confidence': f"{confidence_val * 100:.2f}%",
                    'confidence_value': confidence_val,
                    'confidence_width': f"{confidence_val * 100}%"
                })

            print(f"🎯 Prediction: {results[0]['breed']} ({results[0]['confidence']})")
            return results

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None


classifier = CattleClassifier('../models/trained/best_model.pth')


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not classifier.model_loaded:
        return jsonify({'success': False, 'error': 'Model not ready. Please check if training completed.'})

    try:
        if 'file' in request.files:
            file = request.files['file']
            image = Image.open(io.BytesIO(file.read()))
        elif 'image' in request.form:
            image_data = request.form['image'].split(',')[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        else:
            return jsonify({'success': False, 'error': 'No image provided'})

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        predictions = classifier.predict(image)

        if predictions:
            return jsonify({
                'success': True,
                'predictions': predictions
            })
        else:
            return jsonify({'success': False, 'error': 'Prediction failed - check console for details'})

    except Exception as e:
        print(f"❌ Server error: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})


@app.route('/training-info')
def training_info():
    """Endpoint to get training transparency information"""
    info = {
        'dataset_stats': {
            'total_breeds': 41,
            'total_images': 5928,
            'training_images': 4147,
            'validation_images': 593,
            'test_images': 1186
        },
        'model_performance': {
            'final_accuracy': '52.45%',
            'best_accuracy': '52.45%',
            'training_epochs': 30
        },
        'challenges': [
            'Variable image quality and lighting conditions',
            'Limited samples for rare breeds (as low as 36 images)',
            'High visual similarity between different cattle breeds',
            'Background clutter and varying angles',
            'Dataset imbalance across different breeds'
        ],
        'technical_specs': {
            'model_architecture': 'Vision Transformer (ViT-B/16)',
            'input_size': '224x224 pixels',
            'training_hardware': 'GPU Accelerated',
            'framework': 'PyTorch'
        }
    }
    return jsonify(info)


if __name__ == '__main__':
    print("🚀 Starting BreedAI Cattle Classification Server...")
    print(f"✅ Model loaded: {classifier.model_loaded}")
    print(f"✅ Number of breeds: {len(classifier.breed_names)}")
    app.run(debug=True, host='0.0.0.0', port=5000)