import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

def evaluate_classification(y_true, y_pred, class_names=None):
    """
    Computes classification performance metrics with a focus on macro-F1.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    return {"accuracy": acc, "macro_f1": macro_f1}

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", filepath=None):
    """
    Plots a confusion matrix heatmap using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()

def evaluate_regression(y_true, y_pred):
    """
    Computes standard regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("=== Regression Performance ===")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    return {"mae": mae, "mse": mse, "r2": r2}

def plot_learning_curves(history, metric='loss', title='Learning Curves', filepath=None):
    """
    Plots training and validation metrics over epochs from a Keras training history.
    """
    plt.figure(figsize=(10, 5))
    
    # Check if history is a dictionary or a Keras History object
    hist_dict = history.history if hasattr(history, 'history') else history
    
    epochs = range(1, len(hist_dict[metric]) + 1)
    
    plt.plot(epochs, hist_dict[metric], 'b-', label=f'Training {metric}')
    val_key = f'val_{metric}'
    if val_key in hist_dict:
        plt.plot(epochs, hist_dict[val_key], 'r-', label=f'Validation {metric}')
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()
