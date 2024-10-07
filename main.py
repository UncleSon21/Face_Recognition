import config
from config import load_model
from data_collect import collect_data
from predict import verify, real_time_verification
from preprocess_images import create_dataset
from train import train
from model import siamese_model, L1Dist
from evaluate import evaluate_model
import tensorflow as tf
import numpy as np

def main():
    config.configure_gpu()

    action = input("What would you like to do? (train/evaluate/detect): ").lower().strip()

    if action == 'train':
        collect_option = input("Do you want to collect new data? (yes/no): ").lower().strip() == 'yes'
        if collect_option:
            collect_data()
        train()

    elif action == 'evaluate':
        model = load_model()
        _, test_data = create_dataset()
        recall, precision = evaluate_model(model, test_data)
        print(f"Final Recall: {recall:.4f}")
        print(f"Final Precision: {precision:.4f}")

    elif action == 'detect':
        real_time_verification()

    else:
        print("Invalid action. Please choose train, evaluate, or real-time.")

if __name__ == "__main__":
    main()