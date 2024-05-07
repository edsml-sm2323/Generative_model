import torch

__all__ = ['train_model']


def train_model(model, train_loader, test_loader, criterion,
                optimizer, num_epochs=10, device='cpu'):
    """
    Train a PyTorch model.

    Parameters:
        model: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs (default is 10)
        device: Device to which the model and data should be moved

    Returns:
        print train loss and test loss
    """
    model.to(device)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss1 = []

            outputs = model(inputs)

            for i in range(3):

                loss = criterion(outputs[:, i], targets[:, i])
                loss1.append(loss)

            loss = sum(loss1) / len(loss1)
            # loss = criterion(outputs, targets)
            # loss = -1 * ssim(outputs, targets, data_range=1.0)
            # loss = F.mse_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss1 = []

                outputs = model(inputs)

                for i in range(3):
                    loss = criterion(outputs[:, i], targets[:, i])
                    loss1.append(loss)

                loss = sum(loss1) / len(loss1)
                # loss = criterion(outputs,targets)
                # loss = F.mse_loss(outputs, targets)

                test_loss += loss.item()

        test_loss /= len(test_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], \\'
              f'Training Loss: {train_loss:.4f}, \\'
              f'Test Loss: {test_loss:.4f}')