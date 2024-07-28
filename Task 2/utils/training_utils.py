import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, epochs=50, initial_lr=1e-4, lr_decay_factor=10, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    best_val_loss = float('inf')
    train_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = input.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.size(0)
        epoch_loss = running_loss/len(train_loader.dataset)
        train_history['train_loss'].append(epoch_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100*correct/total
        train_history['val_loss'].append(val_loss)
        train_history['val_accuracy'].append(val_accuracy)

        print(
            f'Epoch [{epoch+1}/{epochs}], Train Loss : {epoch_loss:.4f},Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()

        if (epoch + 1) % 10 == 0:
            for param in optimizer.param_groups:
                param['lr'] /= lr_decay_factor

    model.load_state_dict(best_model_wts)

    return model, train_history
