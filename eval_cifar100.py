import torch
from tqdm.auto import tqdm  # For progress bars


# --- Evaluation on Clean CIFAR-100 Test Set ---
def evaluate_cifar100_test(model, testloader, device):
    """Evaluation on clean CIFAR-100 test set."""
    model.load_state_dict(torch.load("best_model.pth"))  # Load the best model
    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Evaluating on Clean Test Set"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())  # Move predictions to CPU and convert to numpy
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    clean_accuracy = 100.0 * correct / total
    # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    return predictions, clean_accuracy
