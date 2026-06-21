import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# Ethanol example
atoms = {
    0: "C",
    1: "C",
    2: "O",
    3: "H",
    4: "H",
    5: "H",
    6: "H",
    7: "H",
    8: "H"
}

edges = [
    (0, 1),
    (1, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 6),
    (1, 7),
    (2, 8)
]

G.add_edges_from(edges)

pos = nx.spring_layout(G, seed=42)

colors = []

for node in G.nodes():
    atom = atoms[node]

    if atom == "C":
        colors.append("black")
    elif atom == "O":
        colors.append("red")
    else:
        colors.append("lightgray")

plt.figure(figsize=(6, 6))

nx.draw(
    G,
    pos,
    with_labels=True,
    labels=atoms,
    node_color=colors,
    node_size=1200,
    font_color="white"
)

plt.title("Molecular Graph Representation")

plt.savefig("results/molecular_graph.png")
plt.close()

print("Graph saved!")