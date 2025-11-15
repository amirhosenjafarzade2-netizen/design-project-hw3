import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QLabel, QLineEdit, QSpinBox, QGroupBox,
    QFormLayout, QMessageBox, QSplitter, QComboBox, QTextEdit
)
from PyQt5.QtCore import Qt


class BNAMultiBlockApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BNA Multi-Block Histogram Analyzer")
        self.setGeometry(100, 100, 1350, 800)

        self.blocks = []  # List of DataFrames: each block is {'name': str, 'data': pd.DataFrame}
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # === LEFT PANEL ===
        controls = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout()

        # Upload
        upload_btn = QPushButton("Upload .bna File")
        upload_btn.clicked.connect(self.load_bna)
        ctrl_layout.addWidget(upload_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        ctrl_layout.addWidget(self.file_label)

        # Block selector
        block_group = QGroupBox("Select Data Block")
        block_layout = QFormLayout()
        self.block_combo = QComboBox()
        block_layout.addRow("Block:", self.block_combo)
        block_group.setLayout(block_layout)
        ctrl_layout.addWidget(block_group)

        # Column selector (X or Y)
        col_group = QGroupBox("Data Column to Histogram")
        col_layout = QFormLayout()
        self.col_combo = QComboBox()
        self.col_combo.addItems(["Y (Value)", "X (Depth)"])
        col_layout.addRow("Plot:", self.col_combo)
        col_group.setLayout(col_layout)
        ctrl_layout.addWidget(col_group)

        # Chart title
        title_group = QGroupBox("Chart Title")
        t_layout = QFormLayout()
        self.chart_title = QLineEdit("BNA Block Histogram")
        t_layout.addRow("Title:", self.chart_title)
        title_group.setLayout(t_layout)
        ctrl_layout.addWidget(title_group)

        # Bins
        bin_group = QGroupBox("Bins")
        b_layout = QFormLayout()
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 200)
        self.bins_spin.setValue(30)
        b_layout.addRow("Bins:", self.bins_spin)
        bin_group.setLayout(b_layout)
        ctrl_layout.addWidget(bin_group)

        # Axis labels
        axis_group = QGroupBox("Axis Labels")
        a_layout = QFormLayout()
        self.x_label = QLineEdit("Value")
        self.y_label = QLineEdit("Frequency")
        a_layout.addRow("X Label:", self.x_label)
        a_layout.addRow("Y Label:", self.y_label)
        axis_group.setLayout(a_layout)
        ctrl_layout.addWidget(axis_group)

        # X limits
        limit_group = QGroupBox("X-Axis Limits (auto if blank)")
        l_layout = QFormLayout()
        self.x_min = QLineEdit()
        self.x_max = QLineEdit()
        l_layout.addRow("Min:", self.x_min)
        l_layout.addRow("Max:", self.x_max)
        limit_group.setLayout(l_layout)
        ctrl_layout.addWidget(limit_group)

        # Update
        update_btn = QPushButton("Update Histogram")
        update_btn.clicked.connect(self.update_plot)
        update_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        ctrl_layout.addWidget(update_btn)

        ctrl_layout.addStretch()
        controls.setLayout(ctrl_layout)
        controls.setMaximumWidth(380)

        # === RIGHT PANEL: Plot + Preview ===
        right_panel = QSplitter(Qt.Vertical)

        # Plot
        self.figure = plt.Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        right_panel.addWidget(self.canvas)

        # Raw data preview
        self.preview = QTextEdit()
        self.preview.setMaximumHeight(200)
        self.preview.setFontFamily("Courier")
        self.preview.setReadOnly(True)
        right_panel.addWidget(self.preview)

        right_panel.setSizes([600, 200])

        # Main layout
        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(controls)
        main_split.addWidget(right_panel)
        main_split.setSizes([400, 950])
        layout.addWidget(main_split)

        # Style
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #999; border-radius: 6px; margin-top: 10px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; top: -8px; background: white; padding: 0 5px; }
            QLineEdit, QComboBox, QSpinBox { padding: 6px; border: 1px solid #ccc; border-radius: 4px; }
            QPushButton { padding: 8px; border-radius: 4px; }
        """)

    def load_bna(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open BNA", "", "BNA Files (*.bna);;All Files (*)")
        if not path:
            return

        try:
            self.blocks = []
            current_block = None
            block_data = []

            with open(path, 'r') as f:
                lines = f.readlines()

            preview_lines = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('"C",'):
                    # Save previous block
                    if current_block and block_data:
                        df = pd.DataFrame(block_data, columns=['X', 'Y'])
                        self.blocks.append({'name': current_block, 'data': df})
                        block_data = []

                    parts = [p.strip('"') for p in line.split(',')]
                    if len(parts) >= 3:
                        current_block = f"Curve {parts[1]} (n={parts[2]})"
                    else:
                        current_block = f"Curve {len(self.blocks)+1}"

                else:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            block_data.append([x, y])
                        except:
                            continue

                if i < 50:
                    preview_lines.append(line)

            # Save last block
            if current_block and block_data:
                df = pd.DataFrame(block_data, columns=['X', 'Y'])
                self.blocks.append({'name': current_block, 'data': df})

            if not self.blocks:
                raise ValueError("No valid data blocks found.")

            # Update UI
            self.block_combo.clear()
            for b in self.blocks:
                self.block_combo.addItem(b['name'])

            self.file_label.setText(f"<b>{path.split('/')[-1]}</b><br>{len(self.blocks)} blocks, {sum(len(b['data']) for b in self.blocks)} points")
            self.preview.setText("\n".join(preview_lines[-50:]))

            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")

    def update_plot(self):
        if not self.blocks:
            return

        idx = self.block_combo.currentIndex()
        block = self.blocks[idx]
        data = block['data']

        col = self.col_combo.currentText()
        values = data['Y'] if col.startswith("Y") else data['X']
        values = values.dropna()

        bins = self.bins_spin.value()
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # X limits
        try:
            xmin = float(self.x_min.text()) if self.x_min.text().strip() else None
            xmax = float(self.x_max.text()) if self.x_max.text().strip() else None
            range_val = (xmin, xmax) if xmin and xmax else None
        except:
            range_val = None

        ax.hist(values, bins=bins, color='#1f77b4', edgecolor='black', alpha=0.85, range=range_val)
        ax.set_title(block['name'], fontsize=14, fontweight='bold')
        ax.set_xlabel(self.x_label.text())
        ax.set_ylabel(self.y_label.text())
        ax.grid(True, alpha=0.3)

        main_title = self.chart_title.text().strip()
        if main_title:
            self.figure.suptitle(main_title, fontsize=16, fontweight='bold', y=0.98)

        self.figure.tight_layout(rect=[0, 0, 1, 0.94])
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = BNAMultiBlockApp()
    win.show()
    sys.exit(app.exec_())
