# ir_column_annotator_pyside6.py
"""Interactive column‑annotation tool for infrared sequences (PySide6)
--------------------------------------------------------------------

Shortcuts
=========
  n  : save + next frame
  q  : quit (saving current frame first)
  u  : undo last manual column (red)
  p  : toggle auto‑predicted columns (cyan)
  a  : add columns by typing numbers (e.g. "12 77 150")
  r  : remove columns by typing numbers (works for manual OR auto)

Mouse
=====
  • Left‑click  on a *manual* (red) column → remove it
  • Left‑click  on an *auto* (cyan) column → remove that prediction
  • Shift‑click anywhere               → add a manual column at x
"""
from __future__ import annotations

from typing import Dict, List
import sys
import json
from src.edge_detect import EdgeDetector
from src.reading import Reader

import numpy as np
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QSizePolicy,
)


###################################################################################
dataset = Reader('/home/maciejka/Desktop/school/S8/data_challenge/raw_data/Données_CN_V1/GALATEA/GALATEA_C1_REF_N1.h5')  # ← REPLACE by your real reader
method_detection = 'radon' # can be 'histogram', 'radon', or 'canny' - used method to detect edges
method_threshold = 1.3 # threshold when peaks are found - described in sample yaml file
###################################################################################


######################################################################
# 2.  FrameViewer (unchanged except typing tweaks)
######################################################################

_DASHED_CYAN = QPen(QColor("cyan"), 1, Qt.DashLine)
_SOLID_RED = QPen(QColor("red"), 1, Qt.SolidLine)
_TICK_PEN = QPen(QColor("white"), 1, Qt.SolidLine)


class FrameViewer(QWidget):
    column_hovered = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

        self.qimg: QImage | None = None
        self._w = 0; self._h = 0
        self.manual_cols: List[int] = []
        self.pred_cols: List[int] = []
        self.pred_visible = True
        self._undo: List[int] = []

    def set_frame(self, frame: np.ndarray):
        img8 = np.clip(frame / frame.max() * 255, 0, 255).astype(np.uint8)
        self._h, self._w = img8.shape
        self.qimg = QImage(img8.data, self._w, self._h, self._w, QImage.Format_Grayscale8)
        self.update()

    def set_predictions(self, cols: List[int]):
        self.pred_cols = [int(c) for c in cols]; self.update()

    def clear_manual(self):
        self.manual_cols.clear(); self._undo.clear(); self.update()

    def _widget_x_to_col(self, x: float) -> int:
        return int(round(x * self._w / max(1, self.width())))

    def _col_to_x(self, col: int) -> int:
        return int(round(col * self.width() / max(1, self._w)))


    def paintEvent(self, _):  # noqa: N802
        if not self.qimg:
            return
        p = QPainter(self)
        p.drawImage(QRect(0, 0, self.width(), self.height()), self.qimg)
        p.setPen(_TICK_PEN)
        for c in range(0, self._w, 50):
            x = self._col_to_x(c)
            p.drawLine(x, 0, x, 6); p.drawText(x+2, 14, str(c))
        if self.pred_visible:
            p.setPen(_DASHED_CYAN)
            for c in self.pred_cols:
                p.drawLine(self._col_to_x(c), 0, self._col_to_x(c), self.height())
        p.setPen(_SOLID_RED)
        for c in self.manual_cols:
            p.drawLine(self._col_to_x(c), 0, self._col_to_x(c), self.height())

    def mouseMoveEvent(self, ev):  # noqa: N802
        self.column_hovered.emit(self._widget_x_to_col(ev.position().x()))

    def mousePressEvent(self, ev):  # noqa: N802
        if not self.qimg: return
        col = self._widget_x_to_col(ev.position().x())
        if ev.modifiers() & Qt.ShiftModifier:
            self._add(col); return
        if col in self.manual_cols:
            self.manual_cols.remove(col)
        elif col in self.pred_cols:
            self.pred_cols.remove(col)
        else:
            self._add(col)
        self.update()

    def _add(self, col: int):
        if col not in self.manual_cols:
            self.manual_cols.append(col); self._undo.append(col); self.update()

    def undo(self):
        if self._undo:
            c = self._undo.pop();
            if c in self.manual_cols: self.manual_cols.remove(c)
            self.update()

    def toggle_pred(self):
        self.pred_visible = not self.pred_visible; self.update()

    def manual(self) -> List[int]:
        return [int(c) for c in self.manual_cols]


######################################################################
#  MainWindow
######################################################################

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.viewer = FrameViewer(); self.setCentralWidget(self.viewer)
        self.statusBar(); self.viewer.column_hovered.connect(lambda c: self.statusBar().showMessage(f"Column: {c}"))
        self.idx = 0; self.ann: Dict[int, Dict[str, List[int]]] = {}
        self._load()

    def _load(self):
        if self.idx >= len(dataset):
            QMessageBox.information(self, "Done", "All frames annotated ✔"); self.close(); return
        _, img = dataset[self.idx]
        self.viewer.set_frame(img); self.viewer.clear_manual()
        preds = EdgeDetector(img).detect_defects(method_detection, threshold=method_threshold)
        self.viewer.set_predictions(preds)
        self.setWindowTitle(f"IR Column Annotator — Frame {self.idx+1}/{len(dataset)}")

    def _save(self):
        self.ann[int(self.idx)] = {  # ensure key is plain int
            "manual": [int(c) for c in self.viewer.manual()],
            "pred":   [int(c) for c in self.viewer.pred_cols],
        }

    #  keys
    def keyPressEvent(self, ev):  # noqa: N802
        k = ev.key()
        if k == Qt.Key_N:
            self._save(); self.idx += 1; self._load()
        elif k == Qt.Key_Q:
            self._save(); self.close()
        elif k == Qt.Key_U:
            self.viewer.undo()
        elif k == Qt.Key_P:
            self.viewer.toggle_pred()
        elif k == Qt.Key_A:
            self._dialog_add()
        elif k == Qt.Key_R:
            self._dialog_rm()
        else:
            super().keyPressEvent(ev)

    #  dialogs
    @staticmethod
    def _parse(t: str) -> List[int]:
        return [int(tok) for tok in t.replace(',', ' ').split() if tok.isdigit()]

    def _dialog_add(self):
        txt, ok = QInputDialog.getText(self, "Add columns", "Nums (space/comma)")
        if ok and txt.strip():
            for c in self._parse(txt): self.viewer._add(c)

    def _dialog_rm(self):
        txt, ok = QInputDialog.getText(self, "Remove columns", "Nums (space/comma)")
        if not (ok and txt.strip()): return
        cols = set(self._parse(txt))
        self.viewer.manual_cols[:] = [c for c in self.viewer.manual_cols if c not in cols]
        self.viewer.pred_cols[:]   = [c for c in self.viewer.pred_cols   if c not in cols]
        self.viewer.update()

    #  close
    @staticmethod
    def _json_default(o):  # convert any stray NumPy int -> int
        if isinstance(o, np.integer): return int(o)
        raise TypeError

    def closeEvent(self, ev):  # noqa: N802
        if self.ann:
            p, _ = QFileDialog.getSaveFileName(self, "Save annotations", "annotations.json", "JSON (*.json)")
            if p:
                with open(p, "w", encoding="utf-8") as f:
                                        json.dump(self.ann, f, indent=2, default=self._json_default)
        ev.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.resize(960, 700); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
