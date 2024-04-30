import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnext101_32x8d
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to train a model")

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for training (default: 5)')
    parser.add_argument('--device', type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help='Device to use for training (default: cpu)')

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()

    # Variables to use in the script
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DEVICE = args.device


    #### DATA

    transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])


    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


    #### MODEL
    model = resnext101_32x8d().to(device=DEVICE)


    #### TRAIN
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    def train_one_epoch():
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device=DEVICE), labels.to(device=DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch

        return last_loss


    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        model.train(True)
        avg_loss = train_one_epoch()

        print('LOSS train - {}'.format(avg_loss))

