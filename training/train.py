import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import glob


class CattleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class AdvancedViTClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(AdvancedViTClassifier, self).__init__()

        # Load pre-trained Vision Transformer
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Freeze early layers for transfer learning
        for param in list(self.vit.parameters())[:-4]:  # Freeze all but last 4 layers
            param.requires_grad = False

        # Replace classifier head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # Add label smoothing for better generalization
        self.label_smoothing = 0.1

    def forward(self, x, labels=None):
        logits = self.vit(x)

        if labels is not None:
            # Apply label smoothing
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)
            return loss, logits

        return logits


class CattleTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"🔥 GPU: {torch.cuda.get_device_name()}")
            print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

        # Create transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data_splits(self, data_path='data/processed'):
        """Load the preprocessed data splits"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} not found. Run data preprocessor first!")

        # Load breed mapping
        breed_map_path = os.path.join(data_path, 'breed_mapping.json')
        if not os.path.exists(breed_map_path):
            raise FileNotFoundError(f"Breed mapping file not found at {breed_map_path}")

        with open(breed_map_path, 'r') as f:
            breed_map = json.load(f)

        # Convert breed map to list
        breed_names = [breed_map[str(i)] for i in range(len(breed_map))]

        # Load split information
        split_info_path = os.path.join(data_path, 'split_info.json')
        if os.path.exists(split_info_path):
            with open(split_info_path, 'r') as f:
                split_info = json.load(f)
            print(f"📊 Split info: {split_info}")

        # Since we don't have the actual split files saved, we'll recreate them
        # In a real scenario, you'd save the actual file paths
        print("🔄 Recreating dataset structure...")

        # Find all images in the original dataset
        original_data_path = 'data/indian-cattle-breeds'  # Adjust this path
        all_images = []
        all_labels = []

        for breed_idx, breed_name in enumerate(breed_names):
            breed_path = os.path.join(original_data_path, breed_name)
            if os.path.exists(breed_path):
                images = [f for f in os.listdir(breed_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in images:
                    img_path = os.path.join(breed_path, img_file)
                    all_images.append(img_path)
                    all_labels.append(breed_idx)

        print(f"📁 Total images found: {len(all_images)}")

        # Create splits (matching the preprocessor logic)
        from sklearn.model_selection import train_test_split

        # First split: train + temp vs test (20% test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        # Second split: train vs validation (10% of original = 12.5% of temp)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
        )

        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

        print(f"✅ Data splits created:")
        print(f"   Training: {len(X_train)} images")
        print(f"   Validation: {len(X_val)} images")
        print(f"   Test: {len(X_test)} images")

        return splits, breed_names

    def setup_data(self, splits):
        """Setup data loaders"""
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']

        train_dataset = CattleDataset(X_train, y_train, self.train_transform)
        val_dataset = CattleDataset(X_val, y_val, self.val_transform)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,  # Reduced for stability
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,  # Reduced for stability
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"📊 Training samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")

    def setup_model(self, num_classes):
        """Setup model, optimizer, and scheduler"""
        self.model = AdvancedViTClassifier(num_classes=num_classes)
        self.model.to(self.device)

        # Use mixed precision training for GPU
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs']
        )

        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🧠 Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config["epochs"]}')

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if self.device.type == 'cuda' and self.scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    loss, outputs = self.model(images, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, outputs = self.model(images, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss, accuracy

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if self.device.type == 'cuda' and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss, outputs = self.model(images, labels)
                else:
                    loss, outputs = self.model(images, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, accuracy, all_preds, all_labels

    def train(self, splits, breed_names):
        """Main training loop"""
        self.setup_data(splits)
        self.setup_model(len(breed_names))

        best_accuracy = 0
        train_history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

        print("🚀 Starting Training...")
        print("=" * 60)

        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, _, _ = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Save history
            train_history['loss'].append(train_loss)
            train_history['acc'].append(train_acc)
            train_history['val_loss'].append(val_loss)
            train_history['val_acc'].append(val_acc)

            print(f"📈 Epoch {epoch + 1:02d}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_model('best_model.pth', breed_names, train_history)
                print(f"💾 New best model saved with accuracy: {val_acc:.2f}%")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth', breed_names, train_history)

        # Plot training history
        self.plot_training_history(train_history)

        print("🎉 Training completed!")
        print(f"🏆 Best validation accuracy: {best_accuracy:.2f}%")

        return train_history

    def save_model(self, filename, breed_names, history):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': history['val_acc'][-1] if history['val_acc'] else 0,
            'epoch': len(history['val_acc']),
            'breed_names': breed_names,
            'config': self.config
        }

        os.makedirs('models/trained', exist_ok=True)
        torch.save(checkpoint, f'models/trained/{filename}')
        print(f"💾 Model saved as: models/trained/{filename}")

    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        os.makedirs('models/trained', exist_ok=True)
        plt.savefig('models/trained/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Training history plot saved")


def main():
    # Configuration for optimal GPU training
    config = {
        'batch_size': 16,  # Reduced for 41 classes and GPU memory
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 30,  # Reduced for initial testing
        'patience': 10
    }

    print("🐄 Cattle Breed Classification Training")
    print("=" * 50)

    try:
        # Initialize trainer
        trainer = CattleTrainer(config)

        # Load data splits
        splits, breed_names = trainer.load_data_splits()

        print(f"🎯 Training on {len(breed_names)} cattle breeds:")
        for i, breed in enumerate(breed_names):
            print(f"   {i + 1:2d}. {breed}")

        # Start training
        history = trainer.train(splits, breed_names)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you've run the data preprocessor first!")
        print("💡 Command: python utils/data_preprocessor.py --data_path data/indian-cattle-breeds")


if __name__ == "__main__":
    main()