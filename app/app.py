from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
import os
import glob
import numpy as np
from collections import Counter

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


class ModelVotingSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.breed_names = self.load_breed_names()
        self.model_accuracies = {
            'best_model.pth': 52.45,  # From your training output
            'checkpoint_epoch_10.pth': 45.20,  # Estimated
            'checkpoint_epoch_20.pth': 50.80,  # Estimated
            'checkpoint_epoch_30.pth': 52.45  # Same as best_model
        }
        self.load_all_models()

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

    def load_all_models(self):
        model_files = glob.glob('../models/trained/*.pth')
        print(f"🔍 Found {len(model_files)} model files")

        for model_file in model_files:
            model_name = os.path.basename(model_file)
            try:
                checkpoint = torch.load(model_file, map_location=self.device)
                model = AdvancedViTClassifier(num_classes=len(self.breed_names))

                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                model.to(self.device)
                model.eval()

                self.models[model_name] = {
                    'model': model,
                    'accuracy': self.model_accuracies.get(model_name, 45.0),
                    'loaded': True
                }

                print(f"✅ Loaded {model_name} (Accuracy: {self.models[model_name]['accuracy']}%)")

            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                self.models[model_name] = {'loaded': False, 'accuracy': 0.0}

    def predict_ensemble(self, image):
        if not self.models:
            return None

        all_predictions = []
        model_votes = []

        # Get predictions from each model
        for model_name, model_info in self.models.items():
            if not model_info['loaded']:
                continue

            try:
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = model_info['model'](image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    top5_conf, top5_idx = torch.topk(probabilities, 5)

                # Store top 5 predictions from this model
                model_preds = []
                for i in range(5):
                    breed_idx = top5_idx[0][i].item()
                    confidence = top5_conf[0][i].item()
                    breed_name = self.breed_names[breed_idx] if breed_idx < len(
                        self.breed_names) else f"Class_{breed_idx}"

                    model_preds.append({
                        'breed': breed_name,
                        'confidence': confidence,
                        'confidence_percent': f"{confidence * 100:.2f}%"
                    })

                # Add model's top prediction to voting pool (weighted by accuracy)
                weight = model_info['accuracy'] / 100.0  # Convert to 0-1 scale
                top_breed = model_preds[0]['breed']

                # Add votes based on model accuracy (higher accuracy = more votes)
                vote_count = max(1, int(weight * 10))  # 1-10 votes based on accuracy
                model_votes.extend([top_breed] * vote_count)

                all_predictions.append({
                    'model_name': model_name,
                    'accuracy': model_info['accuracy'],
                    'predictions': model_preds,
                    'votes': vote_count,
                    'top_choice': top_breed
                })

            except Exception as e:
                print(f"❌ Prediction error in {model_name}: {e}")
                continue

        if not all_predictions:
            return None

        # Determine final prediction by weighted voting
        vote_counter = Counter(model_votes)
        final_breed, final_votes = vote_counter.most_common(1)[0]
        total_votes = sum(vote_counter.values())

        # Get confidence for the final breed by averaging confidences from models that voted for it
        final_confidence = 0
        contributing_models = 0

        for model_pred in all_predictions:
            for pred in model_pred['predictions']:
                if pred['breed'] == final_breed:
                    final_confidence += pred['confidence']
                    contributing_models += 1
                    break

        final_confidence = final_confidence / contributing_models if contributing_models > 0 else 0

        return {
            'final_prediction': {
                'breed': final_breed,
                'confidence': final_confidence,
                'confidence_percent': f"{final_confidence * 100:.2f}%",
                'votes': final_votes,
                'total_votes': total_votes,
                'vote_percentage': f"{(final_votes / total_votes) * 100:.1f}%"
            },
            'model_predictions': all_predictions,
            'total_models': len(all_predictions),
            'voting_details': dict(vote_counter)
        }


# Initialize the voting system
voting_system = ModelVotingSystem()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            file = request.files['file']
            image = Image.open(io.BytesIO(file.read()))
        else:
            return jsonify({'success': False, 'error': 'No image provided'})

        if image.mode != 'RGB':
            image = image.convert('RGB')

        result = voting_system.predict_ensemble(image)

        if result:
            return jsonify({
                'success': True,
                'result': result
            })
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/model_status')
def model_status():
    model_info = []
    for model_name, model_data in voting_system.models.items():
        model_info.append({
            'name': model_name,
            'loaded': model_data.get('loaded', False),
            'accuracy': model_data.get('accuracy', 0)
        })

    return jsonify({
        'success': True,
        'total_models': len(voting_system.models),
        'loaded_models': sum(1 for m in voting_system.models.values() if m.get('loaded', False)),
        'models': model_info,
        'total_breeds': len(voting_system.breed_names)
    })


if __name__ == '__main__':
    print("🚀 BreedAI Ensemble Voting System Started!")
    print(f"✅ Loaded {sum(1 for m in voting_system.models.values() if m.get('loaded', False))} models")
    print(f"✅ Recognizing {len(voting_system.breed_names)} cattle breeds")
    app.run(debug=True, host='0.0.0.0', port=5000)