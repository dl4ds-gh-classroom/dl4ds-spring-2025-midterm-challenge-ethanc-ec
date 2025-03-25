import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

import wandb


################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Stack 1
        self.conv1_1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stack 2
        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Classifier
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        # Stack 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        # Stack 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        # Classifier
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):
        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        ### TODO - Your code here
        # zero out gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # compute loss
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        running_loss += loss  ### TODO
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100.0 * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval()  # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)  ### TODO -- loss calculation

            running_loss += loss  ### SOLUTION -- add loss from this sample
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100.0 * correct / total})

    val_loss = running_loss / len(valloader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def main():
    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.

    CONFIG = {
        "model": "MyModel",  # Change name when using a different model
        "batch_size": 512,  # run batch size finder to find optimal batch size
        "learning_rate": 0.1,
        "epochs": 5,  # Train for longer in a real scenario
        "num_workers": 4,  # Adjust based on your system
        "device": "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint

    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify)
    ############################################################################

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
            ),  # Example normalization
        ]
    )

    ###############
    # TODO Add validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
    )

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )

    # Split train into train and validation (80/20 split)
    # train_size = ...  ### TODO -- Calculate training set size
    # val_size = ...  ### TODO -- Calculate validation set size
    # trainset, valset = ...  ### TODO -- split into training and validation sets
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    ### TODO -- define loaders and test set
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"]
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"]
    )

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"]
    )

    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = SimpleCNN()  # instantiate your model ### TODO
    model = model.to(CONFIG["device"])  # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size

        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(
            model, trainset, CONFIG["device"], CONFIG["num_workers"]
        )
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()  ### TODO -- define loss criterion
    optimizer = optim.SGD(
        model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9
    )  ### TODO -- define optimizer
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )  # Add a scheduler   ### TODO -- you can optionally add a LR scheduler

    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],  # Log learning rate
            }
        )

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")  # Save to wandb as well

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(
        model, testloader, CONFIG["device"]
    )
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")


if __name__ == "__main__":
    main()
