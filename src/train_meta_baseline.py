# src/train_meta_baseline.py
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Define a PyTorch model for classification
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FakeNewsClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def main():
    print("🔍 Setting up GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load metadata-enhanced CSV
    df = pd.read_csv("data/liar2_meta.csv")

    # 2. Fill NaNs in text fields to avoid vectorizer errors
    df['clean_statement'] = df['clean_statement'].fillna("")
    df['clean_justification'] = df['clean_justification'].fillna("")
    df['context'] = df['context'].fillna("")

    # 3. Prepare features and labels
    X = df.drop(columns=["label"])
    y = df["label"]
    
    # Encode labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    # 4. Split data (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # 5. Define feature transformers for different column types
    # Build the feature extraction pipeline
    feature_pipeline = ColumnTransformer([
        # Text features
        ("stmt_tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), "clean_statement"),
        ("just_tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), "clean_justification"),
        ("context_tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2)), "context"),
        
        # Categorical features with one-hot encoding
        ("meta_cat", OneHotEncoder(handle_unknown="ignore"), ["subject", "speaker", "speaker_description", "state_info"]),
        
        # Numerical features pass through
        ("meta_numerical", "passthrough", [
            "true_counts", "mostly_true_counts", "half_true_counts", 
            "mostly_false_counts", "false_counts", "pants_on_fire_counts"
        ])
    ], sparse_threshold=0.0)  # Force dense output    # 6. Transform the data for PyTorch
    print("Transforming features...")
    X_train_features = feature_pipeline.fit_transform(X_train)
    X_test_features = feature_pipeline.transform(X_test)
    
    # Get the dimensionality of the transformed features
    input_dim = X_train_features.shape[1]
    hidden_dim = 256  # You can adjust this based on your needs
    output_dim = num_classes
    
    print(f"Feature dimensions: {input_dim}")
    
    # 7. Convert data to PyTorch tensors and create DataLoader
    print("Converting to PyTorch tensors...")
    X_train_tensor = torch.FloatTensor(X_train_features)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_features)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Move tensors to GPU
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 8. Initialize model
    print("Initializing model...")
    model = FakeNewsClassifier(input_dim, hidden_dim, output_dim)
    model = model.to(device)  # Move model to GPU
    
    # 9. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 10. Train the model
    print("Starting training...")
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')
    
    print("Training complete!")
    
    # 11. Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_cpu = predicted.cpu().numpy()
        y_test_cpu = y_test_tensor.cpu().numpy()
        
        accuracy = accuracy_score(y_test_cpu, predicted_cpu)
        print(f"✅ Accuracy (with GPU): {accuracy:.4f}")
        print("\n📊 Classification Report:\n", classification_report(y_test_cpu, predicted_cpu))
    
    # 12. Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_pipeline': feature_pipeline,
        'label_encoder': label_encoder
    }, "models/metadata_gpu_model.pt")
    print("Model saved to models/metadata_gpu_model.pt")

if __name__ == "__main__":
    main()
