import torch
from torch.utils.data import DataLoader, TensorDataset
from model.msw_transformer import MSWTransformer
from utils.data_utils import load_ptb_xl
from utils.training_utils import train_model


def main():
    # Load and preprocess dataset
    path = 'data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    X, y, _ = load_ptb_xl(path, sampling_rate=100)

    # Convert dataset to TensorDataset and DataLoader
    dataset = TensorDataset(torch.tensor(
        X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = MSWTransformer(embed_dim=512, num_heads=8, window_sizes=[
                           5, 10, 15], mlp_ratio=4.0, dropout=0.1)

    # Train model
    trained_model, history = train_model(
        model, train_loader, val_loader, num_epochs=50, initial_lr=0.0001, lr_decay_factor=10)


if __name__ == '__main__':
    main()
