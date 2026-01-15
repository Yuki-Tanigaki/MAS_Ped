import osmnx as ox
import matplotlib.pyplot as plt
import contextily as cx

ox.settings.log_console = True
ox.settings.use_cache = True

G = ox.graph_from_address(
    "門司港駅, 北九州市, 福岡県, 日本",
    dist=800,
    network_type="walk"
)

# 3857に投影（contextily向け）
G_3857 = ox.project_graph(G, to_crs="EPSG:3857")

fig, ax = ox.plot_graph(
    G_3857,
    figsize=(10, 10),
    node_size=5,
    node_color="red",
    edge_color="gray",
    edge_linewidth=1,
    bgcolor="white",
    show=False,
    close=False
)

# extent固定
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# basemap（crsを明示）
cx.add_basemap(
    ax,
    source=cx.providers.OpenStreetMap.Mapnik,
    crs="EPSG:3857",
    attribution=False,  # 表示を消したいなら
)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

fig.savefig("mojiko_walk_network_bg.png", dpi=200, bbox_inches="tight")
plt.close(fig)

ox.save_graphml(G, "mojiko_walk.graphml")
