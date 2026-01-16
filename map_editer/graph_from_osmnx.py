#!/usr/bin/env python3
"""
graph_from_osmnx.py

指定した住所(ジオコード)を中心に、OSMnxで徒歩ネットワークを取得して保存する。

例:
  python graph_from_osmnx.py \
    --address "門司港駅, 北九州市, 福岡県, 日本" \
    --dist 800 \
    --network-type walk \
    --out data/mojiko_walk_800.graphml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import osmnx as ox


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a network graph from OSMnx and save it.")
    p.add_argument(
        "--address",
        type=str,
        required=True,
        help='Center address for geocoding (e.g. "門司港駅, 北九州市, 福岡県, 日本")',
    )
    p.add_argument(
        "--dist",
        type=int,
        default=800,
        help="Distance in meters from the address center (default: 800)",
    )
    p.add_argument(
        "--network-type",
        type=str,
        default="walk",
        choices=[
            "walk",
            "drive",
            "drive_service",
            "bike",
            "all",
            "all_private",
        ],
        help="OSMnx network_type (default: walk)",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output file path. Extension should be .graphml (recommended).",
    )
    p.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify graph topology (default: True). Use --no-simplify to disable.",
    )
    p.add_argument(
        "--retain-all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain all components instead of the largest (default: False).",
    )
    p.add_argument(
        "--log-console",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable OSMnx console logging (default: True). Use --no-log-console to disable.",
    )
    p.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable OSMnx caching (default: True). Use --no-cache to disable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ox.settings.log_console = bool(args.log_console)
    ox.settings.use_cache = bool(args.cache)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 取得
    G = ox.graph_from_address(
        args.address,
        dist=args.dist,
        network_type=args.network_type,
        simplify=bool(args.simplify),
        retain_all=bool(args.retain_all),
    )

    # 保存（GraphML）
    # ※ 拡張子が .graphml 以外でも保存はできるが、まずは統一推奨
    ox.save_graphml(G, filepath=str(out_path))

    print("Saved graph:")
    print(f"  nodes: {len(G.nodes):,}")
    print(f"  edges: {len(G.edges):,}")
    print(f"  out  : {out_path.resolve()}")


if __name__ == "__main__":
    main()
