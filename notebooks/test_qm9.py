from torch_geometric.datasets import QM9

dataset = QM9(root="data/QM9")

print("Number of molecules:", len(dataset))

sample = dataset[0]

print(sample)

print("\nNode feature shape:", sample.x.shape)
print("Edge index shape:", sample.edge_index.shape)
print("Target shape:", sample.y.shape)