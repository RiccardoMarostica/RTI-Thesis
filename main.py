import sys
from PyQt6.QtWidgets import QApplication
from gui.mainWindow import MainWindow


def initaliseMainWindow():
    # Build the Application (only one instance can exsists)
    app = QApplication([])
    app.setStyleSheet('.QLabel { font-size: 14pt; }')

    # Show the main window
    mainWindow = MainWindow()

    # And shows it
    sys.exit(app.exec())


if __name__ == "__main__":
    initaliseMainWindow()
    # main()
