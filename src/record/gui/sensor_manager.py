"""
Sensor Manager dock handling IMU/Trigger configuration, connections, and DB settings.

Provides a table to manage devices, load/save config, connect/disconnect devices,
and interact with a SQLite database location.
"""

from record.core.utils import load_config_yml, save_config_yml
from record.gui.utils import MaskDelegate
from record.gui.setup_logger import LOG_DEFAULT_FORMATTER
from record.gui._extanded_qtwidgets import CustomLinedit

from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDockWidget,
    QTableWidgetItem,
    QHeaderView,
    QHBoxLayout,
    QStyledItemDelegate,
    QLineEdit,
)

from record.gui.ui_interfaces.DockWidget_sensor_manager_ui_mod import (
    Ui_DockWidget_IMU_client_manager,
)

from functools import partial
import logging
from record.core.constants import DATABASE_DIR, CONFIG_DIR
from record.core.imu import IMU
from record.core.trigger import Trigger
from sqlalchemy import create_engine

import numpy as np
import time
import yaml


class SensorManagerDockWidget(QDockWidget):
    """Dock widget for configuring and controlling sensors from the GUI."""

    def __init__(self, parent):
        super(SensorManagerDockWidget, self).__init__(parent)
        self.setWindowTitle("Sensor Manager")
        self.setObjectName("sensor_manager")

        # Configuration du logger
        self.setup_logger()

        self.init_params()

        # UI Setups
        self.ui = Ui_DockWidget_IMU_client_manager()
        self.ui.setupUi(self)
        self.init_Ui_Settings()
        self.setWidget(self.ui.dockWidgetContents)

        # Initalizing methods
        self.load_table_from_config_file()

        # Create db engine
        self.create_db_engine()

        # Connection Setups
        self.connections()

    def setup_logger(self):
        """Attach logging handlers for this dock and propagate to parent GUI handler."""
        self.logger = logging.getLogger(f"{SensorManagerDockWidget.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        handler.setFormatter(LOG_DEFAULT_FORMATTER)
        self.logger.addHandler(handler)
        if hasattr(self.parent(), "handler"):
            self.parent().handler.setFormatter(LOG_DEFAULT_FORMATTER)
            self.logger.addHandler(self.parent().handler)

        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    def init_params(self):
        """Initialize parameters, sensor maps, and internal pools."""
        # Attributes
        self.filename_config = f"{CONFIG_DIR}/config.yml"

        # Configuration dictionary for settings which IMUs settings or other
        self.config = ConfigFile(self.filename_config)
        self.Nd = self.config.Nt
        # Dictionary to store IMU objects, key is the name of IMU example : "Shimmer3-0EC4"
        # Only IMUs which are running are kept in this dictionnary
        # If one IMU is closed, it will removed from this dict
        self.imu_poll = {}
        self.trigger_poll = {}
        self.sensor_map = {
            "IMU": {"constructor": IMU, "poll": self.imu_poll},
            "TRIGGER": {"constructor": Trigger, "poll": self.trigger_poll},
        }

        self.checkbox_pool = []
        self.customLineEdit_pool = []
        self.trigger_labels = [
            "imu_right",
            "imu_left",
            "imu_bck",
            "starter",
            "checker",
            "closer",
        ]
        self.combox_label_pool = []
        self.checkbox_selected_pool = []

    def init_Ui_Settings(self):
        """Initial UI configuration and table setup (icons, widgets, defaults)."""
        self.setStyleSheet("QDockWidget::title" "{" "background : darkgray;" "}")
        self.load_ui_icons()

        self.ui.lineEdit_db_name.setText(
            self.config.database["filename"].split("/")[-1]
        )

        self.create_enable_checkboxes()
        self.create_selected_checkboxes()
        self.create_label_comboxes()
        # self.create_customLineEdits()
        self.create_table()

    def load_ui_icons(self):
        """Load toggle icons for DB lock button."""
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("./ui_interfaces/icons/lock_open.png"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.Off,
        )
        icon.addPixmap(
            QtGui.QPixmap("./ui_interfaces/icons/lock_close.png"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.On,
        )
        self.ui.pushButton_db_lock.setIcon(icon)
        self.ui.pushButton_db_lock.setChecked(True)

    def connections(self):
        """Connect UI controls to handler methods."""
        # self.ui.commandLinkButton.clicked.connect(self.establish_imu_connection)
        for i, checkbox in enumerate(self.checkbox_pool):
            checkbox.clicked.connect(partial(self.control_runners, i))

        self.ui.pushButton_connect_selected.clicked.connect(
            self.connect_selected_sensors
        )

        # self.ui.commandLinkButton_close.clicked.connect(self.close_imu_runners)
        self.ui.pushButton_save.clicked.connect(self.save_config_file_from_table)
        self.ui.pushButton_ping_imu.clicked.connect(self.show_imu_data)
        self.ui.pushButton_ping_trigger.clicked.connect(self.show_trigger_data)

        self.ui.pushButton_reset_trigger.clicked.connect(
            lambda: [trigger.reset() for _, trigger in self.trigger_poll.items()]
        )

        # Update self.config attr if any change in table
        self.ui.tableWidget.currentCellChanged.connect(
            self.update_config_attr_from_table
        )

        # database change
        self.ui.pushButton_db_lock.clicked.connect(self.toggle_db_lock)

    def masks_definition(self):
        """Define input masks and set the table delegate for masked columns."""
        self.masks = {
            0: "S\\himmer-XXXX",
            1: "HH:HH:HH:HH:HH:HH",
        }

        delegate = MaskDelegate(self.masks, self.ui.tableWidget)
        self.ui.tableWidget.setItemDelegate(delegate)

    def init_table_cells(self):
        """Initialize each row’s widgets with sensible defaults."""
        init_socket_value = 64344
        for row in range(self.ui.tableWidget.rowCount()):
            for col in range(self.ui.tableWidget.columnCount()):
                column_name = self.ui.tableWidget.horizontalHeaderItem(col).text()
                if column_name == "Status":
                    self.ui.tableWidget.setItem(
                        row,
                        self.columns_names.index("Status"),
                        QTableWidgetItem("Disconnected"),
                    )
                elif column_name == "Socket":
                    spinBox = QtWidgets.QSpinBox(parent=self)
                    spinBox.setMinimum(49152)
                    spinBox.setMaximum(65535)
                    spinBox.setValue(init_socket_value)
                    init_socket_value += 1
                    self.ui.tableWidget.setCellWidget(row, col, spinBox)
                elif column_name == "IP address":
                    comboBox = QtWidgets.QComboBox(parent=self)
                    comboBox.setEditable(True)
                    comboBox.addItems(["127.0.0.1", ""])
                    self.ui.tableWidget.setCellWidget(row, col, comboBox)
                elif column_name == "Enable":
                    # checkBox = QtWidgets.QCheckBox(parent=self)
                    checkBox = self.checkbox_pool[row]
                    checkbox_widget = QtWidgets.QWidget()
                    layout_checkbox = QHBoxLayout(checkbox_widget)
                    layout_checkbox.addWidget(checkBox)
                    layout_checkbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    layout_checkbox.setContentsMargins(0, 0, 0, 0)
                    checkBox.setText(None)
                    self.ui.tableWidget.setCellWidget(row, col, checkBox)
                elif column_name == "Label":
                    comBox = self.combox_label_pool[row]
                    layout_combox = QHBoxLayout(comBox)
                    layout_combox.addWidget(comBox)
                    layout_combox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    layout_combox.setContentsMargins(0, 0, 0, 0)
                    self.ui.tableWidget.setCellWidget(row, col, comBox)
                elif column_name == "Selected":
                    # checkBox = QtWidgets.QCheckBox(parent=self)
                    checkBox = self.checkbox_selected_pool[row]
                    checkbox_widget = QtWidgets.QWidget()
                    layout_checkbox = QHBoxLayout(checkbox_widget)
                    layout_checkbox.addWidget(checkBox)
                    layout_checkbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    layout_checkbox.setContentsMargins(0, 0, 0, 0)
                    checkBox.setText(None)
                    self.ui.tableWidget.setCellWidget(row, col, checkBox)
                # else:
                #     item = QTableWidgetItem()
                #     self.ui.tableWidget.setItem(row, col, item)

    def create_table(self):
        """Build the table header/rows and apply basic header sizing behavior."""
        self.columns_names = [
            "Enable",
            "Status",
            "Selected",
            "Name",
            "IP address",
            "Socket",
            "Label",
        ]

        self.rows_names = self.config.sensors_names

        Nc = len(self.columns_names)
        Nr = self.config.Nt
        self.ui.tableWidget.setColumnCount(Nc)
        self.ui.tableWidget.setRowCount(Nr)
        for i in range(Nc):
            item = QtWidgets.QTableWidgetItem()
            item.setText(self.columns_names[i])
            self.ui.tableWidget.setHorizontalHeaderItem(i, item)

        for i in range(Nr):
            item = QtWidgets.QTableWidgetItem()
            item.setText(self.rows_names[i])
            self.ui.tableWidget.setVerticalHeaderItem(i, item)

        # header = self.ui.tableWidget.horizontalHeader()
        header = self.ui.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

        self.masks_definition()
        self.init_table_cells()

    def create_enable_checkboxes(self):
        """Create and store a checkbox widget per row for the Enable column."""
        for i in range(self.Nd):
            setattr(self.ui, f"checkBox_{i}", QtWidgets.QCheckBox(parent=self))
            self.checkbox_pool.append(getattr(self.ui, f"checkBox_{i}"))

    def create_selected_checkboxes(self):
        """Create and store a checkbox widget per row for the Selected column."""
        for i in range(self.Nd):
            setattr(self.ui, f"checkBox_selected_{i}", QtWidgets.QCheckBox(parent=self))
            self.checkbox_selected_pool.append(
                getattr(self.ui, f"checkBox_selected_{i}")
            )

    def create_label_comboxes(self):
        """Create a label combobox per row populated with known trigger labels."""
        for i in range(self.Nd):
            setattr(self.ui, f"comBox_{i}", QtWidgets.QComboBox(parent=self))
            combox = getattr(self.ui, f"comBox_{i}")
            combox.addItems(self.trigger_labels)
            self.combox_label_pool.append(combox)

    def load_table_from_config_file(self):
        """Populate the table from the YAML config file if present."""
        try:
            self.config = ConfigFile(self.filename_config)

            for index, (key, value) in enumerate(self.config.sensors.items()):
                self.ui.tableWidget.setItem(
                    index,
                    self.columns_names.index("Name"),
                    QTableWidgetItem(key),
                )
                # Ip addrese QCombox
                comboBox_ip_address = self.ui.tableWidget.cellWidget(
                    index,
                    self.columns_names.index("IP address"),
                )
                comboBox_ip_address.clear()
                comboBox_ip_address.addItems([value["ip_address"]])

                # Get spinbox widget from the table
                spinBox = self.ui.tableWidget.cellWidget(
                    index, self.columns_names.index("Socket")
                )
                spinBox.setValue(value["socket"])

                checkBox = self.ui.tableWidget.cellWidget(
                    index, self.columns_names.index("Selected")
                )
                checkBox.setChecked(value["selected"])

                # Fill label column
                combox = self.ui.tableWidget.cellWidget(
                    index, self.columns_names.index("Label")
                )
                # msg = (
                #     f"index: {combox.findText(str(value['label']))} -> {value['label']}"
                # )
                # self.logger.info(msg)
                combox.setCurrentText(value["label"])

        except FileNotFoundError:
            self.logger.error(f"File not found: {self.filename_config}")
            self.logger.error(f"Please select a valid file using the Open button")
            self.logger.error(
                "Or fill the table manually and save it using the Save button"
            )

    def update_config_attr_from_table(self):
        """Sync internal config object from table values."""
        for row in range(self.ui.tableWidget.rowCount()):
            # Extraire le nom de l'appareil
            name_item = self.ui.tableWidget.item(row, self.columns_names.index("Name"))
            name = name_item.text() if name_item else ""

            # Extraire l'adresse IP
            ip_item = self.ui.tableWidget.cellWidget(
                row, self.columns_names.index("IP address")
            )
            # ip_item is a QComboBoxItem, get the text from it
            ip_address = ip_item.currentText() if ip_item else ""

            # Extraire le port du socket via le widget SpinBox
            socket_widget = self.ui.tableWidget.cellWidget(
                row, self.columns_names.index("Socket")
            )
            socket = socket_widget.value() if socket_widget else 0

            # Extraire l'état selected
            checkbox_selected = self.ui.tableWidget.cellWidget(
                row, self.columns_names.index("Selected")
            )
            selected = checkbox_selected.isChecked()

            # Extraire la valeur de label données
            label_widget = self.ui.tableWidget.cellWidget(
                row, self.columns_names.index("Label")
            )
            label = label_widget.currentText()

            # Ajouter les données extraites au dictionnaire sous le format requis
            self.config.add_sensor(
                name,
                {
                    "ip_address": ip_address,
                    "socket": socket,
                    "label": label,
                    "selected": selected,
                },
            )

        self.logger.info("self.config attribute updated successfully.")

    def save_config_file_from_table(self):
        """Save the configuration from the table to the config file."""

        self.update_config_attr_from_table()

        # Sauvegarder le dictionnaire dans le fichier YAML
        try:
            self.config.dump_config_file(path=self.filename_config)
            self.logger.info(
                f"Configuration sauvegardée avec succès dans {self.filename_config}"
            )
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du fichier : {e}")

    def control_runners(self, ind):
        if self.checkbox_pool[ind].isChecked():
            self.establish_connection(row_index=ind)
        else:
            self.close_runners(row_index=ind)

    def establish_connection(self, row_index):
        name = self.ui.tableWidget.item(
            row_index, self.columns_names.index("Name")
        ).text()

        if self.checkbox_pool[
            row_index
        ].isChecked():  # Correct way to check if a checkbox is checked
            ip_address = self.ui.tableWidget.cellWidget(
                row_index, self.columns_names.index("IP address")
            ).currentText()
            socket = self.ui.tableWidget.cellWidget(
                row_index, self.columns_names.index("Socket")
            ).value()
            label = self.ui.tableWidget.cellWidget(
                row_index, self.columns_names.index("Label")
            ).currentText()

            # Assuming you have a method to establish the connection
            try:
                for key, value in self.sensor_map.items():
                    if key in name.upper():
                        value["poll"][name] = value["constructor"](
                            name, ip_address=ip_address, port=socket, label=label
                        )
                        if hasattr(self.parent(), "handler"):
                            value["poll"][name].logger.addHandler(self.parent().handler)
                        value["poll"][name].connect()
                        time.sleep(0.5)
                        if not value["poll"][name].is_connected:
                            raise ConnectionError(
                                "Unable to connect, check if device is on and IP address/socket are correct"
                            )
                        value["poll"][name].record_setting(self.engine)
                        break
            except Exception as e:
                self.logger.error(f"{e}")
                status_text = "Failed to connect"
                self.checkbox_pool[row_index].setChecked(False)
            else:
                status_text = "Connected"

            self.ui.tableWidget.setItem(
                row_index,
                self.columns_names.index("Status"),
                QTableWidgetItem(status_text),
            )

    def close_runners(self, row_index):
        name = self.ui.tableWidget.item(
            row_index, self.columns_names.index("Name")
        ).text()

        for key, value in self.sensor_map.items():
            if key in name.upper():
                if (
                    name in value["poll"].keys()
                    and not self.checkbox_pool[row_index].isChecked()
                ):  # Correct way to check if a checkbox is checked
                    success = value["poll"][name].close()

                    if success:
                        status_text = "Disconnected"
                    else:
                        status_text = "Failed to disconnect"

                    self.ui.tableWidget.setItem(
                        row_index,
                        self.columns_names.index("Status"),
                        QTableWidgetItem(status_text),
                    )
                    del value["poll"][name]

    def closeEvent(self, event):
        for i in range(self.ui.tableWidget.rowCount()):
            self.close_runners(i)
        event.accept()
        # self.logger.info("\033[31mTODO\033[0m")

    def show_imu_data(self):
        self.logger.info("Checking...")
        if not self.imu_poll.items():
            self.logger.info("No IMU connected !")
        else:
            for k, v in self.imu_poll.items():
                # self.logger.info(f"{k}: \t{v.ping_data()}")
                v.ping_data()

    def show_trigger_data(self):
        self.logger.info("Checking...")
        if not self.trigger_poll.items():
            self.logger.info("No trigger connected !")
        else:
            for k, v in self.trigger_poll.items():
                # self.logger.info(f"{k}: \t{v.ping_data()}")
                v.ping_data()

    def create_db_engine(self):
        """Create a SQLAlchemy engine using the selected DB filename in the UI."""
        db_name = self.ui.lineEdit_db_name.text()
        self.engine = create_engine(f"sqlite:///{DATABASE_DIR}/{db_name}")

    def toggle_db_lock(self):
        """Toggle DB edit/lock state and (re)create engine when locking."""
        if self.ui.pushButton_db_lock.isChecked():
            self.logger.info("Creating new database file")
            self.create_db_engine()
        else:
            self.logger.info("Editing database file name...")

    def connect_selected_sensors(self):
        """Connect or disconnect all rows whose Selected checkbox is checked."""
        for row in range(self.ui.tableWidget.rowCount()):
            checkbox_selected = self.ui.tableWidget.cellWidget(
                row, self.columns_names.index("Selected")
            )
            selected = checkbox_selected.isChecked()
            if selected:
                if self.ui.pushButton_connect_selected.isChecked():
                    self.checkbox_pool[row].setChecked(True)
                    self.establish_connection(row)
                else:
                    self.checkbox_pool[row].setChecked(False)
                    self.close_runners(row_index=row)


class ConfigFile:
    """Lightweight wrapper around the GUI YAML configuration content."""

    def __init__(self, path: str):
        self.__dict__.update(load_config_yml(path))

    def add_sensor(self, sensor_name: str, sensor_data: dict):
        """Add or update a sensor in the internal config dict-of-lists structure."""
        if sensor_name in self.sensors_names:
            # Update value of an existing sensor
            for sensor_list in (self.imus, self.triggers):
                for sensor in sensor_list:
                    if sensor_name in sensor:
                        sensor[sensor_name].update(sensor_data)
                        return
        else:
            # Add a new sensor entry
            new_sensor = {sensor_name: sensor_data}
            if "IMU" in sensor_name.upper():
                # Append sensor to imus list attribute
                self.imus.append(new_sensor)
            elif "TRIGGER" in sensor_name.upper():
                # Append sensor to triggers list attribute
                self.triggers.append(new_sensor)

    def dump_config_file(self, path: str = None):
        """Save current config to the specified path."""
        save_config_yml(path, self.__dict__)

    @property
    def sensors(self):
        """
        Return all sensors as a flat dict {name: settings}.
        """
        sensors = {}
        for key, value in self.__dict__.items():
            if key.startswith("imus") or key.startswith("triggers"):
                for element in value:
                    sensors.update(element)
        return sensors

    @property
    def sensors_names(self):
        return list(self.sensors.keys())

    @property
    def N_imu(self):
        """
        Return number of registered IMU sensors.
        """
        return len(self.imus)

    @property
    def N_trigger(self):
        """
        Return number of registered trigger sensors.
        """
        return len(self.triggers)

    @property
    def Nt(self):
        """
        Return total number of sensors registered.
        """
        return self.N_imu + self.N_trigger


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    DockWidget = SensorManagerDockWidget(None)
    DockWidget.show()
    DockWidget.resize(750, 300)
    sys.exit(app.exec())
    # from record.core.constants import MODULE_PATH

    # config_file_path = f"{CONFIG_DIR}/config.yml"
    # config = ConfigFile(path=config_file_path)
