
import sys
from PyQt5.QtWidgets import QApplication

from qspec.models._gui._ui._mainUi import MainUi


def main():
    app = QApplication(sys.argv)
    main_window = MainUi()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
