"""
Main application window for the record-imucap GUI.

Sets up dock widgets for sensor management, sequence control, IGUS manager,
and a console log, wiring up basic interactions.
"""

import sys
import logging


from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt

from record.gui.sensor_manager import SensorManagerDockWidget
from record.gui.log_consol import ConsoleLogDockWidget
from record.gui.sequence import SequenceDockWidget
from record.gui._motor import IgusManagerDockWidget
from record.gui.setup_logger import LOG_DEFAULT_FORMATTER
from record.gui.ui_interfaces.MainWindow_ui_mod import Ui_MainWindow


class MainWindow(QMainWindow):
    """Top-level window orchestrating all GUI dock widgets."""

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_logger()
        self.init_ui_settings()
        self.init_parameters()
        self.connections()
        self.welcome_msg()

    def init_ui_settings(self):
        """Configure window state, dock widgets and placements."""
        # Maximize mainwindow
        self.showMaximized()

        # Hidden widget
        self.ui.centralwidget.hide()

        # Dock Widgets
        self.create_dock_widgets()

        # Sensor Manager
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.ui.dockWidget_sensor_manager
        )

        # Sequence
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.ui.dockWidget_sequence
        )

        # Console
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.ui.dockWidget_consoleLog
        )

        self.tabifyDockWidget(
            self.ui.dockWidget_sensor_manager, self.ui.dockWidget_igus
        )

    def init_parameters(self):
        """Initialize window-level parameters or state placeholders."""

    def connections(self):
        """Wire up top-level signal/slot connections (placeholder)."""

    def setup_logger(self):
        """Create the console dock and attach a stream handler to logger."""
        # Console Log
        self.ui.dockWidget_consoleLog = ConsoleLogDockWidget(self)
        self.ui.dockWidget_consoleLog.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

        self.handler = self.ui.dockWidget_consoleLog.ui.textEdit_console
        self.logger = logging.getLogger(f"{SensorManagerDockWidget.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        handler.setFormatter(LOG_DEFAULT_FORMATTER)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    def create_dock_widgets(self):
        """Instantiate dock widgets and configure allowed areas."""
        # Sensor Manager
        self.ui.dockWidget_sensor_manager = SensorManagerDockWidget(self)
        self.ui.dockWidget_sensor_manager.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )

        # Sequence
        self.ui.dockWidget_sequence = SequenceDockWidget(self)
        self.ui.dockWidget_sequence.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Igus
        self.ui.dockWidget_igus = IgusManagerDockWidget(self)

    def welcome_msg(self):
        """Log basic app info and version to the console."""
        from importlib.metadata import version

        self.logger.info(f"Welcome to record-imucap GUI (version: {version('record')})")
        self.logger.info("Authored by: L.MARECHAL, C.ELMO")
        self.logger.info("Licensing: MIT Licence")


if __name__ == "__main__":
    """Run the GUI application with custom app icon."""
    from PyQt6 import QtGui, QtCore
    from record.gui.utils import UiLoader
    import record

    # UiLoader()
    module_folder = record.__path__[0]
    icons_folder = module_folder + "/gui/ui_interfaces/icons"

    app = QApplication(sys.argv)
    app_icon = QtGui.QIcon()
    app_icon.addFile(f"{icons_folder}/icon_app.png", QtCore.QSize(16, 16))
    app_icon.addFile(f"{icons_folder}/icon_app.png", QtCore.QSize(24, 24))
    app_icon.addFile(f"{icons_folder}/icon_app.png", QtCore.QSize(32, 32))
    app_icon.addFile(f"{icons_folder}/icon_app.png", QtCore.QSize(48, 48))
    app_icon.addFile(f"{icons_folder}/icon_app.png", QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
