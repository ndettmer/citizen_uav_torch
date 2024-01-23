from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QPalette, QPixmap
from PySide6.QtWidgets import QLabel, QLineEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget

from citizenuav.data import InatDataModule
from citizenuav.io import get_pid_from_path, store_split_inat_metadata


class SelectorWidget(QWidget):
    def __init__(self, data_dir: Union[str, Path], species: Optional[Union[list[str], str]] = None):
        super().__init__()

        self.data_dir = data_dir
        self.dm = InatDataModule(data_dir, species, img_size=512)
        self.ds = self.dm.ds
        self.metadata = self.dm.metadata
        self.current_index = 0

        if "hand_picked" not in self.dm.metadata.columns:
            self.metadata["hand_picked"] = pd.Series(index=self.metadata.index, dtype=bool)
            self.metadata.hand_picked = None
        else:
            # If we re-enter the process, start at the first sample that is not set yet.
            nan_pids = self.metadata[self.metadata.hand_picked.isna()].index.values
            for i in range(len(self.ds)):
                pid = get_pid_from_path(self.ds.samples[i][0])
                if pid in nan_pids:
                    self.current_index = i
                    break

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        self.class_text = QLineEdit()
        self.class_text.setReadOnly(True)

        self.accept_button = QPushButton("Accept (A)")
        accept_palette = self.accept_button.palette()
        accept_palette.setColor(QPalette.Button, QColor(0, 255, 0))
        self.accept_button.setPalette(accept_palette)
        self.accept_button.setShortcut(QKeySequence(Qt.Key_A))
        self.accept_button.clicked.connect(self.accept_image)

        self.decline_button = QPushButton("Decline (D)")
        decline_palette = self.decline_button.palette()
        decline_palette.setColor(QPalette.Button, QColor(255, 0, 0))
        self.decline_button.setPalette(decline_palette)
        self.decline_button.setShortcut(QKeySequence(Qt.Key_D))
        self.decline_button.clicked.connect(self.decline_image)

        self.back_button = QPushButton("Back (Backspace)")
        self.back_button.setShortcut(QKeySequence(Qt.Key_Backspace))
        self.back_button.clicked.connect(self.go_back)

        self.start_from_scratch_button = QPushButton("Start from scratch")
        start_from_scratch_palette = self.start_from_scratch_button.palette()
        start_from_scratch_palette.setColor(QPalette.Button, QColor(200, 0, 0))
        self.start_from_scratch_button.setPalette(start_from_scratch_palette)
        self.start_from_scratch_button.clicked.connect(self.start_from_scratch)

        self.save_button = QPushButton("Save (Ctrl + S)")
        self.save_button.setShortcut(QKeySequence(Qt.CTRL, Qt.Key_S))
        self.save_button.clicked.connect(self.save_action)

        self.progress_bar = QProgressBar()

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.class_text)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.accept_button)
        self.layout.addWidget(self.decline_button)
        self.layout.addWidget(self.back_button)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.start_from_scratch_button)
        self.layout.addWidget(self.save_button)

        self.show_image()

    def _get_current_pid(self):
        path, _ = self.ds.samples[self.current_index]
        return get_pid_from_path(path)

    def _get_current_class(self):
        pid = self._get_current_pid()
        row = self.metadata.loc[pid]
        return row.species

    def _create_current_text(self):
        pid = self._get_current_pid()
        sample = self.metadata.loc[pid]
        class_text = sample.species
        if sample.hand_picked is True:
            status_text = ", accepted"
        elif sample.hand_picked is False:
            status_text = ", declined"
        else:
            status_text = ""

        text = f"{pid}, {class_text}{status_text}"

        if "distance" in sample:
            text = text + f", {np.around(sample.distance, 2)} m"
        return text

    def show_image(self):
        self.progress_bar.setValue(self.current_index / len(self.ds) * 100)
        self.progress_bar.setFormat(f"{self.current_index}/{len(self.ds)}")
        self.class_text.setText(self._create_current_text())
        pixmap = QPixmap(self.ds.samples[self.current_index][0])
        pixmap = pixmap.scaled(self.label.size(), aspectMode=Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)

    def _go_steps(self, delta):
        future_index = self.current_index + delta
        if 0 <= future_index <= len(self.ds):
            self.current_index = future_index
            self.show_image()

    def next_image(self):
        if not self.current_index % 100:
            store_split_inat_metadata(self.metadata, self.data_dir)
        self._go_steps(1)

    def go_back(self):
        self._go_steps(-1)

    def _pick_image(self, value: bool):
        pid = self._get_current_pid()
        self.metadata.loc[pid, "hand_picked"] = value
        self.next_image()

    def accept_image(self):
        self._pick_image(True)

    def decline_image(self):
        self._pick_image(False)

    def start_from_scratch(self):
        self.metadata.hand_picked = None
        self.current_index = 0
        self.show_image()

    def __del__(self):
        if hasattr(self, "data_dir") and hasattr(self, "metadata"):
            store_split_inat_metadata(self.metadata, self.data_dir)

    def save_action(self):
        store_split_inat_metadata(self.metadata, self.data_dir)
