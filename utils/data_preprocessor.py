import os
import shutil
import random

import json
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse


class CattleDataPreprocessor:
    def __init__(self, data_path, output_path, img_size=224):
        self.data_path = data_path
        self.output_path = output_path
        self.img_size = img_size
        self.breeds = []

    def discover_breeds(self):
        """Discover all cattle breeds from the dataset"""
        if os.path.exists(self.data_path):
            self.breeds = [d for d in os.listdir(self.data_path)
                           if os.path.isdir(os.path.join(self.data_path, d))]
            self.breeds.sort()
            print(f"🐄 Found {len(self.breeds)} cattle breeds: {self.breeds}")
        return self.breeds

    def check_image(self, img_path):
        """Check if image is valid"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except:
            return False

    def process_images(self):
        """Process and validate all images"""
        valid_images = []
        labels = []

        for breed_idx, breed in enumerate(self.breeds):
            breed_path = os.path.join(self.data_path, breed)
            if os.path.exists(breed_path):
                images = [f for f in os.listdir(breed_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                breed_images = []
                for img_file in images:
                    img_path = os.path.join(breed_path, img_file)
                    if self.check_image(img_path):
                        breed_images.append(img_path)

                valid_images.extend(breed_images)
                labels.extend([breed_idx] * len(breed_images))
                print(f"✅ {breed}: {len(breed_images)} valid images")

        return valid_images, labels

    def create_splits(self, test_size=0.2, val_size=0.1):
        """Create train/validation/test splits"""
        images, labels = self.process_images()

        # First split: train + temp vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Second split: train vs validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )

        print(f"\n📊 Dataset Split Summary:")
        print(f"🏋️  Training: {len(X_train)} images")
        print(f"⚖️  Validation: {len(X_val)} images")
        print(f"🧪 Test: {len(X_test)} images")

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'breeds': self.breeds
        }

    def save_splits(self, splits):
        """Save splits to files"""
        # ... existing code ...

        # Save the actual splits with file paths
        split_data = {
            'train_images': splits['train'][0],
            'train_labels': splits['train'][1],
            'val_images': splits['val'][0],
            'val_labels': splits['val'][1],
            'test_images': splits['test'][0],
            'test_labels': splits['test'][1],
        }

        with open(os.path.join(self.output_path, 'splits.json'), 'w') as f:
            json.dump(split_data, f, indent=2)

        print(f"💾 Splits saved to {self.output_path}")

    # def save_splits(self, splits):
    #     """Save splits to files"""
    #     # Create output directory
    #     os.makedirs(self.output_path, exist_ok=True)
    #
    #     # Save breed mapping
    #     breed_map = {i: breed for i, breed in enumerate(splits['breeds'])}
    #     with open(os.path.join(self.output_path, 'breed_mapping.json'), 'w') as f:
    #         import json
    #         json.dump(breed_map, f, indent=2)
    #
    #     # Save split information
    #     split_info = {
    #         'train_size': len(splits['train'][0]),
    #         'val_size': len(splits['val'][0]),
    #         'test_size': len(splits['test'][0]),
    #         'total_classes': len(splits['breeds'])
    #     }
    #
    #     with open(os.path.join(self.output_path, 'split_info.json'), 'w') as f:
    #         json.dump(split_info, f, indent=2)
    #
    #     print(f"💾 Splits saved to {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description='Cattle Data Preprocessor')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to raw dataset')
    parser.add_argument('--output_path', type=str, default='data/processed',
                        help='Output path for processed data')

    args = parser.parse_args()

    preprocessor = CattleDataPreprocessor(args.data_path, args.output_path)
    breeds = preprocessor.discover_breeds()

    if not breeds:
        print("❌ No breeds found! Check your data path.")
        return

    splits = preprocessor.create_splits()
    preprocessor.save_splits(splits)


if __name__ == "__main__":
    main()
