# model_train.py

# Import necessary libraries
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
from ultralytics import YOLO

# Create YAML configuration file for YOLOv8
def create_yaml_file():
    data = {
        'train': r"C:\Users\vaibh\Desktop\Aish\train",  # Use raw string to avoid escape issues
        'val': r"C:\Users\vaibh\Desktop\Aish\valid",    # Use raw string to avoid escape issues
        'nc': 1,
        'names': ['pothole']
    }
    yaml_path = 'C:\\Users\\vaibh\\Desktop\\Aish\\data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    print(f"YAML file created at {yaml_path}")

# Train YOLOv8 model
def train_model():
    model = YOLO('yolov8n.pt')
    model.train(data='C:\\Users\\vaibh\\Desktop\\Aish\\data.yaml',
            epochs=10,  # Reduced epochs for quicker testing
            imgsz=416,  # Reduced image size
            lrf=0.2,
            momentum=0.9,
            batch=32,
            project='C:\\Users\\vaibh\\Desktop\\Aish',
            name='experiment2',
            half=True)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No Pothole', 'Predicted Pothole'],
                yticklabels=['Actual No Pothole', 'Actual Pothole'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

# Plot precision, recall, and F1 score
def plot_precision_recall_f1(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
    plt.plot(thresholds, recall[:-1], label='Recall', color='orange')
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score', color='green')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score')
    plt.legend()
    plt.grid()
    plt.show()

# Main function to run the entire workflow
def main():
    create_yaml_file()
    train_model()

    # Example true and predicted labels for confusion matrix
    y_true = [0, 1, 1, 0, 1, 1, 0, 1]  # Replace with your true labels
    y_pred = [0, 1, 0, 0, 1, 1, 0, 1]  # Replace with your predicted labels
    plot_confusion_matrix(y_true, y_pred)

    # Example true labels and predicted probabilities for precision-recall
    y_scores = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.6, 0.4, 0.75]  # Replace with your model scores
    plot_precision_recall_f1(y_true, y_scores)

if __name__ == "__main__":
    main()
