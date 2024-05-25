import matplotlib.pyplot as plt
import os
import json

def plot_training_history(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)

    # Extracting training history
    loss = history['loss']
    val_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # Plotting training and validation loss
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_validation_loss.png')

    # Plotting training and validation accuracy
    plt.figure()
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_validation_accuracy.png')

    # Show the plots
    plt.show()

if __name__ == "__main__":
    history_path = 'training_history.json'  # Change this to the actual path of your training history file
    plot_training_history(history_path)
