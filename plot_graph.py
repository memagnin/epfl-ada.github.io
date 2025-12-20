import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

# load edges from csv
csv_path = "gamegraph.csv"
df = pd.read_csv(csv_path).dropna(subset=["linkSource", "linkTarget"])
df["linkSource"] = df["linkSource"].astype(str).str.strip()
df["linkTarget"] = df["linkTarget"].astype(str).str.strip()

# build directed graph
G = nx.from_pandas_edgelist(df, source="linkSource", target="linkTarget", create_using=nx.DiGraph())


# Datastory is in red
red_nodes = {
    "Wikipedia graph structure",
    "Introduction",
    "Power of the first links",
    "General player behaviour",
    "Lazy players"
}

backbone_order = [
    "Introduction",
    "Wikipedia graph structure",
    "Power of the first links",
    "General player behaviour",
    "Lazy players"
]

red_edges = {
    ("Introduction", "Wikipedia graph structure"),
    ("Wikipedia graph structure", "Power of the first links"),
    ("Power of the first links", "General player behaviour"),
    ("General player behaviour", "Lazy players"),
}

# 4) colors
node_colors = ["red" if n in red_nodes else "lightgray" for n in G.nodes()]
edge_colors = ["red" if (u, v) in red_edges else "gray" for (u, v) in G.edges()]
edge_widths = [3.0 if (u, v) in red_edges else 1.2 for (u, v) in G.edges()]

pos = {}
for i, n in enumerate(backbone_order):
    pos[n] = (i * 100, 0) 

# Layout
pos = nx.spring_layout(
    G,
    pos=pos,
    fixed=backbone_order,
    seed=42,
    k=0.2, # increase to spread things out
    iterations=300
)
# pos = nx.kamada_kawai_layout(G, pos=pos)

plt.figure(figsize=(12, 8))

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, edgecolors="black", linewidths=0.6)
nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_colors,
    width=edge_widths,
    arrows=G.is_directed(),
    arrowsize=18,
    connectionstyle="arc3,rad=0.08"  # slight curve helps readability
)
nx.draw_networkx_labels(G, pos, font_size=8)

plt.axis("off")
plt.tight_layout()
plt.savefig("assets/img/gamegraph.png", dpi=300)
plt.show()