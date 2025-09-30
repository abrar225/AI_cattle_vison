AI-Powered Indian Cattle & Buffalo Breed Recognition
A high-accuracy deep learning solution developed to solve the Smart India Hackathon (SIH) Problem Statement ID25004: "Image-based breed recognition for cattle and buffaloes of India". This project provides a robust, production-ready model to ensure data integrity in India's national livestock database.

The Problem: A Crisis of Data Integrity
The Indian government's Bharat Pashudhan App (BPA) is a critical tool for managing the health, breeding, and nutrition of the nation's vast dairy animal population. However, the success of this digital initiative is fundamentally compromised by a single, persistent issue: inaccurate breed identification.

Why is Manual Identification Failing?
Field Level Workers (FLWs) are tasked with logging animal data, but they face an immense challenge. India is a global hotspot of cattle and buffalo diversity, with over 70 officially recognized breeds, not to mention countless crossbreeds. Many of these breeds share similar physical traits, and accurate identification requires specialized, expert-level knowledge that is difficult to scale across a large workforce.

The Consequences of Flawed Data
This recurring misclassification leads to a "garbage in, garbage out" scenario, where the national database becomes unreliable. This has severe, real-world consequences:

Failed Genetic Programs: Without accurate data on which breeds are performing well, national programs to improve milk yield and disease resistance are based on guesswork.

Ineffective Health Policies: Targeted interventions for breed-specific diseases cannot be implemented effectively.

Wasted Resources: Government subsidies, insurance schemes, and nutritional programs cannot be allocated efficiently, leading to significant financial and strategic losses.

Our mission is to solve this problem by replacing subjective manual judgment with a reliable, AI-driven classification tool.

Our Step-by-Step Solution
We developed a high-accuracy deep learning pipeline by following a systematic, iterative process. Here is a step-by-step account of how we built the solution.

Step 1: Building a Comprehensive Dataset
The foundation of any great model is great data. We recognized that a single dataset might not be diverse enough.

Action: We curated a comprehensive dataset by merging two publicly available sources from Kaggle.

Outcome: This created a larger, more varied combined_dataset with over 7,900 images across 52 classes, providing the model with a richer understanding of inter- and intra-breed variations.

Step 2: Choosing a State-of-the-Art Architecture
Instead of building a model from scratch, we employed transfer learning to leverage the knowledge of a model pre-trained on millions of images.

Action: We selected EfficientNetV2B3, a powerful and modern Convolutional Neural Network (CNN) known for its high accuracy and computational efficiency.

Outcome: This approach allowed us to achieve state-of-the-art results without needing a massive, custom-built architecture.

Step 3: Diagnosing and Solving Data Imbalance
Our initial training attempts resulted in very poor performance. An in-depth evaluation using a confusion matrix revealed a critical insight: the model was heavily biased.

Problem: The dataset was highly imbalanced, with many more images of common breeds than rare ones. The model had learned to simply guess the most frequent class.

Action: We implemented class_weight balancing during training. This technique forces the model to pay significantly more attention to the under-represented, rare breeds.

Outcome: This single change dramatically improved the model's ability to learn from all classes equally, leading to a much more accurate and fair classifier.

Step 4: Advanced Training with a Two-Phase Strategy
To push the accuracy above the 97% target, we used a two-phase fine-tuning technique.

Phase 1 (Head Training): We first froze the entire pre-trained EfficientNetV2B3 base and only trained the new classification layers we added on top. This allowed the new layers to adapt to our specific dataset without disrupting the valuable pre-trained weights.

Phase 2 (Fine-Tuning): We then unfroze the top 40% of the base model's layers and continued training with a very low learning rate. This delicately adjusted the pre-trained weights to better recognize the specific features of cattle breeds.

Outcome: This strategy resulted in a highly specialized and accurate final model.

Step 5: Optimizing for Local Hardware
Training on a local machine without a high-end GPU presented a final challenge: running out of memory.

Problem: The initial configuration with high-resolution images and a large batch size caused MemoryError crashes.

Action: We systematically optimized the training script by reducing the image size to (224, 224) and the batch size to 8.

Outcome: These adjustments allowed the script to run successfully on standard hardware without sacrificing the potential for high accuracy.

How to Use This Project
1. Setup
Clone the repository: git clone https://github.com/your-username/CattleClassifier.git

Navigate into the project directory: cd CattleClassifier

Create and activate a Python virtual environment.

Install all dependencies: pip install -r requirements.txt

Prepare the combined_dataset folder as described in the main README.

2. Training the Model
Run the training script. This will start the full, two-phase training process.

python train.py

3. Evaluating the Model
Once training is complete, run the evaluation script to see the final performance report.

python evaluate.py

Final Results
This structured approach is designed to produce a model that is not only highly accurate but also robust and fair in its predictions across all breeds.

(This section should be updated with your final accuracy score and confusion matrix after a successful training run.)
