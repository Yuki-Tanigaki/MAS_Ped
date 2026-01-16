# MAP_Editer
```bash 
uv run python map_editer/graph_from_osmnx.py \
  --address "門司港駅, 北九州市, 福岡県, 日本" \
  --dist 800 \
  --network-type walk \
  --out data/mojiko_walk_800.graphml
```

```bash 
uv run python map_editer/map_editor.py --in data/mojiko_walk_800.graphml
```

背景地図でエラーになる / 重いときは：
```bash 
uv run python map_editer/map_editor.py --in data/mojiko_walk_800.graphml --no-basemap
```