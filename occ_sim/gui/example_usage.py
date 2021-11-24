import sys
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCore import QFile, Slot
from ui_mainwindow import Ui_MainWindow

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, FigureCanvas
from matplotlib.figure import Figure

import numpy as np

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=16, height=9, dpi=200):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.mpl_canvas = MplCanvas(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.refresh_button.clicked.connect(self.say_hello)
        self.ui.sim_combobox.currentTextChanged.connect(self.on_ui_change)
        self.ui.options_stack.setEnabled(False)

        self.static_canvas = FigureCanvas(Figure(figsize=(16,9)))
        self.ax = self.static_canvas.figure.subplots()
        self.line, = self.ax.plot(list(range(5)), np.random.rand(5))
        self.ui.mpl_layout_space.addWidget(self.static_canvas)


    @Slot()
    def on_ui_change(self):
        # 0 = blank (should always be inactive)
        # 1 = NoisyMM
        # 2 = SingleCue
        if self.ui.sim_combobox.currentText() == "Noisy MM":
            self.ui.options_stack.setEnabled(True)
            self.ui.options_stack.setCurrentIndex(1)
        elif self.ui.sim_combobox.currentText() == "Single Cue":
            self.ui.options_stack.setEnabled(True)
            self.ui.options_stack.setCurrentIndex(2)
        else:
            self.ui.options_stack.setEnabled(False)
            self.ui.options_stack.setCurrentIndex(0)

    @Slot()
    def say_hello(self):
        print("say_hello: click")
        self.ax.cla()
        self.ax.set_title(self.ui.conf_light_azimuth_spinbox.value())
        self.ax.plot(list(range(5)), np.random.rand(5))
        self.static_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
