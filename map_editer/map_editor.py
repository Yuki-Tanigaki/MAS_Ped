#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import osmnx as ox
import networkx as nx

from pyproj import Transformer

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------------------
# Utility: projection and distance
# ----------------------------
def point_to_segment_distance(px, py, x1, y1, x2, y2) -> float:
    """Distance from point P to segment AB (in data coordinates)."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return float(np.hypot(px - x1, py - y1))
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return float(np.hypot(px - x2, py - y2))
    b = c1 / c2
    bx, by = x1 + b * vx, y1 + b * vy
    return float(np.hypot(px - bx, py - by))


def geom_distance_point_to_polyline(px, py, xs: np.ndarray, ys: np.ndarray) -> float:
    """Min distance from point to polyline."""
    if len(xs) < 2:
        return float(np.hypot(px - xs[0], py - ys[0])) if len(xs) == 1 else float("inf")
    dmin = float("inf")
    for i in range(len(xs) - 1):
        d = point_to_segment_distance(px, py, xs[i], ys[i], xs[i + 1], ys[i + 1])
        if d < dmin:
            dmin = d
    return dmin


# ----------------------------
# Selection types
# ----------------------------
@dataclass
class Selection:
    kind: str  # "node" or "edge"
    node_id: Optional[int] = None
    edge_key: Optional[Tuple[int, int, int]] = None  # (u, v, key)


# ----------------------------
# Main Window
# ----------------------------
class MapEditor(QMainWindow):
    def __init__(self, graph_path: Path, out_path: Optional[Path] = None, show_basemap: bool = True):
        super().__init__()
        self.setWindowTitle("GraphML Map Editor (viewer + attribute editor)")

        self.graph_path = graph_path
        self.out_path = out_path or graph_path

        # Load graph
        self.G: nx.MultiDiGraph = ox.load_graphml(graph_path)

        # Build projection transformer: WGS84 -> WebMercator
        self.to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Extract projected node coordinates
        self.node_ids: List[int] = list(self.G.nodes)
        xs, ys = [], []
        for nid in self.node_ids:
            data = self.G.nodes[nid]
            lon = data.get("x", None)
            lat = data.get("y", None)
            if lon is None or lat is None:
                xs.append(np.nan)
                ys.append(np.nan)
            else:
                x2, y2 = self.to_3857.transform(float(lon), float(lat))
                xs.append(x2)
                ys.append(y2)
        self.node_x = np.array(xs, dtype=float)
        self.node_y = np.array(ys, dtype=float)

        # Prebuild edge polylines in 3857 (for click selection)
        self.edges_index: List[Tuple[int, int, int]] = []
        self.edges_xy: List[Tuple[np.ndarray, np.ndarray]] = []
        for u, v, k, edata in self.G.edges(keys=True, data=True):
            # If geometry exists, use it; else use straight segment between nodes
            geom = edata.get("geometry", None)
            if geom is not None:
                try:
                    # shapely LineString: coords in lon/lat
                    coords = list(geom.coords)
                    lons = np.array([c[0] for c in coords], dtype=float)
                    lats = np.array([c[1] for c in coords], dtype=float)
                    xs, ys = self.to_3857.transform(lons, lats)
                    xs = np.array(xs, dtype=float)
                    ys = np.array(ys, dtype=float)
                except Exception:
                    xs = np.array([self.node_x[self.node_ids.index(u)], self.node_x[self.node_ids.index(v)]], dtype=float)
                    ys = np.array([self.node_y[self.node_ids.index(u)], self.node_y[self.node_ids.index(v)]], dtype=float)
            else:
                # straight
                try:
                    iu = self.node_ids.index(u)
                    iv = self.node_ids.index(v)
                    xs = np.array([self.node_x[iu], self.node_x[iv]], dtype=float)
                    ys = np.array([self.node_y[iu], self.node_y[iv]], dtype=float)
                except ValueError:
                    continue

            self.edges_index.append((u, v, k))
            self.edges_xy.append((xs, ys))

        # UI
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Top toolbar
        topbar = QHBoxLayout()
        self.lbl_file = QLabel(f"Input: {self.graph_path}  /  Output: {self.out_path}")
        self.btn_save = QPushButton("保存（GraphML上書き）")
        self.btn_save_as = QPushButton("別名で保存…")
        topbar.addWidget(self.lbl_file, 1)
        topbar.addWidget(self.btn_save)
        topbar.addWidget(self.btn_save_as)
        layout.addLayout(topbar)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # Left: Map panel (matplotlib)
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        splitter.addWidget(self.canvas)

        # Right: Info panel (attributes table)
        info = QWidget()
        info_layout = QVBoxLayout(info)
        self.lbl_selected = QLabel("選択：なし（ノード/エッジをクリック）")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["key", "value"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Buttons for attribute ops
        btnrow = QHBoxLayout()
        self.btn_add_attr = QPushButton("属性行を追加")
        self.btn_apply_attr = QPushButton("この選択に反映")
        btnrow.addWidget(self.btn_add_attr)
        btnrow.addWidget(self.btn_apply_attr)

        info_layout.addWidget(self.lbl_selected)
        info_layout.addWidget(self.table, 1)
        info_layout.addLayout(btnrow)

        splitter.addWidget(info)
        splitter.setSizes([900, 450])

        # State
        self.selection: Optional[Selection] = None
        self.show_basemap = show_basemap

        # Signals
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.btn_add_attr.clicked.connect(self.on_add_attr_row)
        self.btn_apply_attr.clicked.connect(self.on_apply_attr_to_graph)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_save_as.clicked.connect(self.on_save_as)

        # Draw initial
        self.redraw()

    # ----------------------------
    # Drawing
    # ----------------------------
    def redraw(self):
        self.ax.clear()

        # edges
        for (xs, ys) in self.edges_xy:
            self.ax.plot(xs, ys, linewidth=1)

        # nodes
        self.ax.scatter(self.node_x, self.node_y, s=8)

        # highlight selection
        if self.selection:
            if self.selection.kind == "node" and self.selection.node_id is not None:
                try:
                    i = self.node_ids.index(self.selection.node_id)
                    self.ax.scatter([self.node_x[i]], [self.node_y[i]], s=40)
                except ValueError:
                    pass
            elif self.selection.kind == "edge" and self.selection.edge_key is not None:
                try:
                    idx = self.edges_index.index(self.selection.edge_key)
                    xs, ys = self.edges_xy[idx]
                    self.ax.plot(xs, ys, linewidth=3)
                except ValueError:
                    pass

        self.ax.set_aspect("equal", adjustable="box")

        # optional basemap
        if self.show_basemap:
            try:
                import contextily as cx
                cx.add_basemap(self.ax, crs="EPSG:3857")
            except Exception:
                # basemap is optional
                pass

        self.ax.set_title("Click node/edge to select")
        self.canvas.draw_idle()

    # ----------------------------
    # Selection
    # ----------------------------
    def _tol_data_units(self, tol_px: float = 10.0) -> float:
        """Convert pixel tolerance to data coordinate tolerance (approx)."""
        # if not ready
        if self.ax is None or self.canvas is None:
            return 10.0
        inv = self.ax.transData.inverted()
        x0, y0 = inv.transform((0, 0))
        x1, y1 = inv.transform((tol_px, 0))
        return float(np.hypot(x1 - x0, y1 - y0))

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        px, py = float(event.xdata), float(event.ydata)
        tol = self._tol_data_units(12.0)

        # 1) nearest node
        dx = self.node_x - px
        dy = self.node_y - py
        d2 = dx * dx + dy * dy
        i = int(np.nanargmin(d2))
        d_node = float(np.sqrt(d2[i]))

        # 2) nearest edge (polyline distance)
        d_edge = float("inf")
        best_edge_idx = -1
        for ei, (xs, ys) in enumerate(self.edges_xy):
            d = geom_distance_point_to_polyline(px, py, xs, ys)
            if d < d_edge:
                d_edge = d
                best_edge_idx = ei

        # Decide selection: prefer node if it's close enough and closer than edge
        if d_node <= tol and d_node <= d_edge:
            node_id = self.node_ids[i]
            self.selection = Selection(kind="node", node_id=node_id)
            self.show_attributes_for_selection()
            self.redraw()
            return

        if d_edge <= tol and best_edge_idx >= 0:
            self.selection = Selection(kind="edge", edge_key=self.edges_index[best_edge_idx])
            self.show_attributes_for_selection()
            self.redraw()
            return

        # nothing selected
        self.selection = None
        self.lbl_selected.setText("選択：なし（ノード/エッジをクリック）")
        self.table.setRowCount(0)
        self.redraw()

    # ----------------------------
    # Attributes table
    # ----------------------------
    def show_attributes_for_selection(self):
        if not self.selection:
            return

        attrs: Dict[str, Any] = {}
        title = ""

        if self.selection.kind == "node" and self.selection.node_id is not None:
            nid = self.selection.node_id
            attrs = dict(self.G.nodes[nid])
            title = f"選択：Node {nid}"
        elif self.selection.kind == "edge" and self.selection.edge_key is not None:
            u, v, k = self.selection.edge_key
            attrs = dict(self.G.edges[u, v, k])
            title = f"選択：Edge (u={u}, v={v}, key={k})"

        self.lbl_selected.setText(title)

        # Fill table
        self.table.setRowCount(0)
        for r, (k, v) in enumerate(sorted(attrs.items(), key=lambda kv: str(kv[0]))):
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.table.setItem(r, 1, QTableWidgetItem("" if v is None else str(v)))

    def on_add_attr_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(""))
        self.table.setItem(r, 1, QTableWidgetItem(""))

    def _read_table_as_dict(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for r in range(self.table.rowCount()):
            k_item = self.table.item(r, 0)
            v_item = self.table.item(r, 1)
            k = (k_item.text() if k_item else "").strip()
            v = (v_item.text() if v_item else "")
            if not k:
                continue
            out[k] = v
        return out

    def on_apply_attr_to_graph(self):
        if not self.selection:
            QMessageBox.information(self, "Info", "ノードまたはエッジを選択してください。")
            return

        new_attrs = self._read_table_as_dict()

        if self.selection.kind == "node" and self.selection.node_id is not None:
            nid = self.selection.node_id
            # Update node attrs
            for k, v in new_attrs.items():
                self.G.nodes[nid][k] = v

        elif self.selection.kind == "edge" and self.selection.edge_key is not None:
            u, v, k = self.selection.edge_key
            for kk, vv in new_attrs.items():
                self.G.edges[u, v, k][kk] = vv

        QMessageBox.information(self, "Info", "属性を反映しました。")

    # ----------------------------
    # Save
    # ----------------------------
    def on_save(self):
        try:
            ox.save_graphml(self.G, filepath=str(self.out_path))
            QMessageBox.information(self, "Saved", f"保存しました:\n{self.out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as GraphML", str(self.out_path), "GraphML (*.graphml)")
        if not path:
            return
        self.out_path = Path(path)
        self.lbl_file.setText(f"Input: {self.graph_path}  /  Output: {self.out_path}")
        self.on_save()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input GraphML path")
    ap.add_argument("--out", dest="out_path", default=None, help="Output GraphML path (default: overwrite input)")
    ap.add_argument("--no-basemap", action="store_true", help="Disable basemap tiles")
    return ap.parse_args()


def main():
    args = parse_args()
    app = QApplication([])
    w = MapEditor(
        graph_path=Path(args.in_path),
        out_path=Path(args.out_path) if args.out_path else None,
        show_basemap=not args.no_basemap,
    )
    w.resize(1400, 900)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
