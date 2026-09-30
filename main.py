import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow, TORCH_AVAILABLE

import PyQt5
qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qt_plugin_path, 'platforms')

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font = QFont('Microsoft YaHei', 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()