import torch

__all__ = ["train_model"]


def train_model(
    model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device="cpu"
):
    model.to(device)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "best_model.pth")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], \\"
            f"Training Loss: {train_loss:.4f}, \\"
            f"Test Loss: {test_loss:.4f}"
        )
