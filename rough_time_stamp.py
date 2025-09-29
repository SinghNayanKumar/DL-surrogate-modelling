import torch

try:
    scalar_stats = torch.load('data/trns/gnn_stats_x_scalar.pt')
    onehot_stats = torch.load('data/gnn_stats_x_onehot.pt')

    print("--- Scalar Stats (Specialist) ---")
    print("Mean shape:", scalar_stats['mean_x'].shape) # Should be torch.Size([14])
    print("Std shape: ", scalar_stats['std_x'].shape)  # Should be torch.Size([14])

    print("\n--- One-Hot Stats (Generalist) ---")
    print("Mean shape:", onehot_stats['mean_x'].shape) # Should be torch.Size([16])
    print("Std shape: ", onehot_stats['std_x'].shape)  # Should be torch.Size([16])

except FileNotFoundError as e:
    print(e)