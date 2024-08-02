
import sys
from PyQt5.QtWidgets import QApplication

from qspec.models._gui._ui._mainUi import MainUi


def main(db_path=None, data_path=''):
    app = QApplication(sys.argv)
    main_window = MainUi(db_path=db_path, data_path=data_path)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
