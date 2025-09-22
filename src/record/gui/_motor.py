"""
Dock widget to control Igus motors for test automation and calibration.

Provides connection, movement, velocity control, and scripted test runs
such as non-linear and hysteresis sequences, with IMU recording hooks.
"""

import glob
import time

from enum import Enum

from igus import core as igc
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOption,
    QStyleOptionComboBox,
    QStyleOptionMenuItem,
    QStyleOptionViewItem,
    QStylePainter,
)

from record.gui.ui_interfaces.dock_igus_manager_ui_mod import Ui_IgusManager
from record.gui.setup_logger import LOG_DEFAULT_FORMATTER
import logging


class IgusManagerDockWidget(QtWidgets.QDockWidget):
    """GUI dock for managing two Igus devices (linear/rotate)."""

    def __init__(self, parent=None):
        super(IgusManagerDockWidget, self).__init__(parent)
        self.setWindowTitle("Igus Manager")
        self.setObjectName("IgusManager")
        self.ui = Ui_IgusManager()
        self.ui.setupUi(self)
        self.setup_logger()
        self.setWidget(self.ui.dockWidget_IgusManagerContents)
        self.init_Parameters()
        self.init_Ui_Settings()
        self.connections()
        self.update()

    def setup_logger(self):
        """Configure logger and attach to parent’s console handler when available."""
        self.logger = logging.getLogger(f"{IgusManagerDockWidget.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        handler.setFormatter(LOG_DEFAULT_FORMATTER)
        self.logger.addHandler(handler)
        if hasattr(self.parent(), "handler"):
            self.handler = self.parent().handler
            self.handler.setFormatter(LOG_DEFAULT_FORMATTER)
            self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    def init_Ui_Settings(self):
        """Initialize ports combo boxes and populate test types."""
        self.setStyleSheet("QDockWidget::title" "{" "background : darkgray;" "}")
        # ports = glob.glob("/dev/tty[A-Za-z]*")
        # ports_usb = glob.glob("/dev/ttyUSB*")
        # ports = ports + ports_usb
        ports = igc.serial_ports()
        self.ui.comboBox_port_linear.addItems(ports)
        self.ui.comboBox_port_rotate.addItems(ports)
        self.ui.comboBox_port_rotate.setCurrentIndex(
            self.ui.comboBox_port_rotate.count() - 1
        )

        # Add test type
        tests_list = [i.value for i in TestType]
        self.ui.comboBox_test_type.clear()
        self.ui.comboBox_test_type.addItems(tests_list)

    def init_Parameters(self):
        """Initialize Igus instances bucket and absolute position tracking."""
        self.igus_bucket = {"linear": None, "rotate": None}
        self.abs_pos = {"linear": 0, "rotate": 0}

        self.igus_test_handler = IgusTestHandler(self)

    def connections(self):
        """Connect UI controls to device and test actions."""
        # mode linear
        self.ui.checkBox_run_linear.clicked.connect(
            lambda: self.create_igus(mode="linear", widget=self.ui.checkBox_run_linear)
        )
        # set velocity
        self.ui.spinBox_velocity_linear.editingFinished.connect(
            lambda: self.set_velocity(
                mode="linear", value=self.ui.spinBox_velocity_linear.value()
            )
        )

        self.ui.pushButton_Move_linear.clicked.connect(
            lambda: self.move_igus(mode="linear")
        )
        self.ui.pushButton_Home_linear.clicked.connect(
            lambda: self.home_igus(mode="linear")
        )
        self.ui.pushButton_Stop_linear.clicked.connect(
            lambda: self.stop_igus(mode="linear")
        )
        self.ui.pushButton_refresh_pos_linear.clicked.connect(
            lambda: self.refresh_pos(mode="linear")
        )
        self.ui.pushButton_debug_linear.clicked.connect(
            lambda: self.debug_igus(mode="linear")
        )
        self.ui.pushButton_zero_linear.clicked.connect(
            lambda: self.go_zero(mode="linear")
        )

        # mode rotate
        self.ui.checkBox_run_rotate.clicked.connect(
            lambda: self.create_igus(mode="rotate", widget=self.ui.checkBox_run_rotate)
        )
        # set velocity
        self.ui.spinBox_velocity_rotate.editingFinished.connect(
            lambda: self.set_velocity(
                mode="rotate", value=self.ui.spinBox_velocity_rotate.value()
            )
        )

        self.ui.pushButton_Move_rotate.clicked.connect(
            lambda: self.move_igus(mode="rotate")
        )
        self.ui.pushButton_Home_rotate.clicked.connect(
            lambda: self.home_igus(mode="rotate")
        )
        self.ui.pushButton_Stop_rotate.clicked.connect(
            lambda: self.stop_igus(mode="rotate")
        )
        self.ui.pushButton_refresh_pos_rotate.clicked.connect(
            lambda: self.refresh_pos(mode="rotate")
        )
        self.ui.pushButton_debug_rotate.clicked.connect(
            lambda: self.debug_igus(mode="rotate")
        )
        self.ui.pushButton_zero_rotate.clicked.connect(
            lambda: self.go_zero(mode="rotate")
        )

        self.ui.pushButton_refresh_device.clicked.connect(
            lambda: self.update_device_list()
        )

        self.ui.pushButton_Check_move.clicked.connect(
            lambda x: print("On move: ", self.check_on_move())
        )

        # IGUS TEST HANDLER
        self.ui.pushButton_run_test.clicked.connect(
            lambda: self.igus_test_handler.start()
        )

    def update(self):
        """Placeholder for periodic UI/device updates."""
        pass

    def update_device_list(self):
        """Refresh available serial ports and repopulate combo boxes."""
        # ports = glob.glob("/dev/tty[A-Za-z]*")
        ports = ports = igc.serial_ports()
        self.ui.comboBox_port_linear.clear()
        self.ui.comboBox_port_rotate.clear()
        self.ui.comboBox_port_linear.addItems(ports)
        self.ui.comboBox_port_rotate.addItems(ports)
        self.ui.comboBox_port_rotate.setCurrentIndex(
            self.ui.comboBox_port_rotate.count() - 1
        )

    def create_igus(self, mode, widget):
        """Instantiate an Igus object for selected port and set initial velocity."""
        wid_suffix = widget.objectName().split("_")[-1]
        igus = igc.IGUS(
            port=str(
                self.findChild(QComboBox, f"comboBox_port_{wid_suffix}").currentText()
            )
        )
        igus.logger.addHandler(self.handler)
        time.sleep(0.5)
        igus.set_mode(mode)
        self.igus_bucket[mode] = igus
        time.sleep(0.5)
        self.set_velocity(
            mode=mode,
            value=self.findChild(QSpinBox, f"spinBox_velocity_{mode}").value(),
        )

    def set_velocity(self, mode, value):
        """Set maximum velocity for the specified axis (linear/rotate)."""
        if self.igus_bucket[mode] is not None:
            self.igus_bucket[mode].set_max_velocity(value)

    def move_igus(self, mode, value=None, unit=None, direction=None):
        """Move the selected axis by value, in unit (step/rev/deg) and direction."""
        if value is None:
            value = self.findChild(
                QDoubleSpinBox, f"doubleSpinBox_value_{mode}"
            ).value()
            unit = str(
                self.findChild(QComboBox, f"comboBox_value_unit_{mode}").currentText()
            )
            direction = str(
                self.findChild(QComboBox, f"comboBox_dir_{mode}").currentText()
            )

        self.igus_bucket[mode].move_position(
            value=value, unit=unit, direction=direction
        )

        value_step = (
            value if unit == "step" else round(value / self.igus_bucket[mode].CONST_REV)
        )

        if direction == "R":
            self.abs_pos[mode] += value_step
        else:
            self.abs_pos[mode] -= value_step

    def home_igus(self, mode):
        """Home the specified axis and reset absolute position to zero."""
        self.igus_bucket[mode].move_home()
        self.abs_pos[mode] = 0

    def go_zero(self, mode):
        """Return the axis to absolute zero if not already there."""
        if self.abs_pos[mode] != 0:
            self.igus_bucket[mode].move_position(
                value=self.abs_pos[mode], unit="step", direction="L"
            )
            self.abs_pos[mode] = 0
        else:
            print("Motor already at zero")

    def stop_igus(self, mode):
        """Stop motion for the specified axis."""
        self.igus_bucket[mode].move_stop()

    def debug_igus(self, mode):
        """Enable internal pass_check flag for debugging movements."""
        self.igus_bucket[mode].pass_check = True

    def refresh_pos(self, mode):
        """Read and display current position using the selected unit widget."""
        pos = self.igus_bucket[mode].get_current_position(
            unit=str(
                self.findChild(
                    QComboBox, f"comboBox_currentPos_unit_{mode}"
                ).currentText()
            )
        )

        label_widget = self.findChild(QLabel, f"label_current_pos_{mode}")

        label_widget.setText(str(pos))

    def check_on_move(self):
        """Return True if any Igus device is currently moving."""
        bool_list = [
            value.on_move
            for key, value in self.igus_bucket.items()
            if value is not None
        ]

        return any(bool_list)


class TestType(Enum):
    NLINEAR = "NLINEAR"
    HYSTERESIS = "HYSTERESIS"


class IgusTestHandler(QThread):
    """Background runner to execute Igus test sequences and control IMUs."""

    def __init__(self, parent):
        super(IgusTestHandler, self).__init__(parent)

        self.setup_logger()

        self.imu_poll = self.parent().parent().ui.dockWidget_sensor_manager.imu_poll

    def setup_logger(self):
        """Configure logger and attach to parent’s console handler when available."""
        self.logger = logging.getLogger(f"{IgusTestHandler.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        handler.setFormatter(LOG_DEFAULT_FORMATTER)
        self.logger.addHandler(handler)
        if hasattr(self.parent(), "handler"):
            self.handler = self.parent().handler
            self.handler.setFormatter(LOG_DEFAULT_FORMATTER)
            self.logger.addHandler(self.handler)

        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    def run(self):
        """Entry point: dispatch to selected test type (NLINEAR/HYSTERESIS)."""

        # Get current test type
        self.test_type = TestType(
            str(self.parent().ui.comboBox_test_type.currentText())
        )

        if self.test_type == TestType.NLINEAR:
            self.nlinear_test()
        elif self.test_type == TestType.HYSTERESIS:
            self.hysteresis_test()
        else:
            raise ValueError("Invalid test type")

    @property
    def name_offset(self):
        """Starting index for generated test IDs."""
        return int(self.parent().ui.spinBox_name_offset.value())

    @property
    def N_repeat(self):
        """Number of repetitions to execute for the test."""
        return self.parent().ui.spinBox_repeat.value()

    @property
    def step_deg(self):
        """Angular step size in degrees."""
        return self.parent().ui.doubleSpinBox_step_angle.value()

    @property
    def target_angle(self):
        """Target angle in degrees for the test."""
        # unit deg
        return self.parent().ui.doubleSpinBox_target_angle.value()

    @property
    def step(self):
        """Step size in motor steps (converted from degrees)."""
        return self.step_deg * 400 / 360  # Conversion from deg to steps

    @property
    def N_step(self):
        """Number of steps derived from target angle and step size."""
        return int(self.target_angle / self.step_deg) + 1

    def nlinear_test(self):
        """Run non-linear test: incremental rotations with recording."""
        self.logger.info("RUNNING NO LINEAR TEST")

        for i in range(self.N_repeat):
            self.logger.info(f"REPEAT N°: {i}")
            self.test_id = self.get_test_id(i)

            for j in range(self.N_step):
                self.parent().move_igus("rotate", self.step, "step", "R")
                while self.parent().check_on_move():
                    time.sleep(0.1)

                time.sleep(1)
                self.start_record()
                time.sleep(1)
                self.stop_record()

            self.logger.info("Back to HOME")
            self.parent().move_igus("rotate", self.N_step * self.step, "step", "L")
            while self.parent().check_on_move():
                time.sleep(0.1)

    def hysteresis_test(self):
        """Run hysteresis test: forward then backward rotations with recording."""
        self.logger.info("RUNNING HYSTERESIS TEST")

        for i in range(self.N_repeat):
            self.test_id = self.get_test_id(i)
            for j in range(self.N_step):
                self.parent().move_igus("rotate", self.step, "step", "R")
                while self.parent().check_on_move():
                    time.sleep(0.1)

                time.sleep(1)
                self.start_record()
                time.sleep(1)
                self.stop_record()

            while self.parent().check_on_move():
                time.sleep(0.1)

            for j in range(self.N_step):
                self.parent().move_igus("rotate", self.step, "step", "L")
                while self.parent().check_on_move():
                    time.sleep(0.1)

                time.sleep(1)
                self.start_record()
                time.sleep(1)
                self.stop_record()

    def start_record(self):
        """Start IMU recording for the current generated test ID."""
        [imu.record_start(test_id=self.test_id) for _, imu in self.imu_poll.items()]

    def stop_record(self):
        """Stop IMU recording if currently active."""
        [imu.record_stop() for _, imu in self.imu_poll.items() if imu.is_recording]

    def get_test_id(self, i):
        """Compose a test_id string from type, target and step, with an index."""
        i += self.name_offset
        basename = f"{self.test_type.value}_{self.target_angle}_{self.step_deg}"
        test_id = basename + f"_{i}"
        self.parent().ui.lineEdit.setText(test_id)
        return test_id


if __name__ == "__main__":
    import sys
    import time

    message = (
        "Which class to test ?\n"
        + "1 -> IgusManagerDockWidget\n"
        + "Enter corresponding number: "
    )

    resp = input(message)

    app = QtWidgets.QApplication(sys.argv)
    if resp == "1":
        window = IgusManagerDockWidget()
    window.show()
    sys.exit(app.exec_())
