"""
Sequence control dock: orchestrates tests, triggers, recording, and plots.

Includes helpers to build the VTK scene, manage test plans, handle trigger
roles, record sensor streams, and replay plots from the database.
"""

from record.gui.ui_interfaces.DockWidget_sequence_ui_mod import Ui_DockWidget_sequence
from enum import Enum
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import QThread, QEvent
from PyQt6.QtWidgets import (
    QDockWidget,
    QTableWidgetItem,
    QStyledItemDelegate,
    QComboBox,
    QLineEdit,
)
from PyQt6.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
import pyqtgraph.exporters

from record.core.geometry import Quaternion, signed_angle
from record.core.decorators import threaded, execute_once
from record.core.imu import IMU
from record.core.trigger import Trigger
from record.core.database import TestDB, check_for_existing_test_id, delete_by_test_id
from record.core.test import TestStatus, Tests, Test, Patients
import record.core.database as db
import record.core.postprocessing as pp
from record.gui.setup_logger import LOG_DEFAULT_FORMATTER
from record.gui._extanded_vtk import SceneVTKQT
from record.gui._extanded_qtwidgets import CheckableComboBox, show_popup, error_popup
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkCommonMath import vtkQuaternion
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

import logging
import numpy as np
from time import sleep
from datetime import datetime

signed_angle_comp = signed_angle

import warnings

warnings.simplefilter("ignore", ResourceWarning)

import record

MODULE_FOLDER = record.__path__[0]
ICONS_FOLDER = MODULE_FOLDER + "/gui/ui_interfaces/icons"


class CustomComboBox(QComboBox):
    """ComboBox supporting per-item data objects and styled rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items_data = {}

    # def mouseMoveEvent(self, a0):
    #     return super().mouseMoveEvent(a0)
    def mousePressEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.MouseButtonPress:
            self.update_lines_styles()
        return super().mousePressEvent(event)

    def addItemsFromDict(self, items: dict):
        """Add items from a dict where text=key and userData=value."""
        for key, value in items.items():
            self.addItem(
                key, value
            )  # Ajouter les éléments avec la valeur en tant que donnée utilisateur

    def addItem(self, text, userData=None):
        """Insert an item as a QStandardItem with optional data payload."""
        item = QtGui.QStandardItem()
        item.setText(text)
        if not userData is None:
            item.setData(userData)

        self.model().appendRow(item)

    def update_lines_styles(self):
        """Update background color based on attached TestStatus in item data."""

        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.data() is not None:
                if item.data().status == TestStatus.PASSED:
                    item.setBackground(QtGui.QColor("#69FF8C"))
                elif item.data().status == TestStatus.FAILED:
                    item.setBackground(QtGui.QColor("#BF4267"))
                else:
                    item.setBackground(QtGui.QColor("#E0E0E0"))


class SequenceDockWidget(QDockWidget):
    """Dock widget coordinating test sequences and visualization."""

    def __init__(self, parent):
        super(SequenceDockWidget, self).__init__(parent)

        # UI Setups
        self.ui = Ui_DockWidget_sequence()
        self.ui.setupUi(self)
        self.init_Ui_Settings()
        self.setup_logger()
        self.init_parameters()

        # Connection Setups
        self.connections()

    def setup_logger(self):
        """Configure logger and attach to parent’s console handler when available."""
        self.logger = logging.getLogger(f"{SequenceDockWidget.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        handler.setFormatter(LOG_DEFAULT_FORMATTER)
        self.logger.addHandler(handler)
        if hasattr(self.parent(), "handler"):
            self.handler = self.parent().handler
            self.handler.setFormatter(LOG_DEFAULT_FORMATTER)
            self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    def init_parameters(self):
        """Initialize references to sensor pools, engine, and handlers."""
        # Sensors parameters
        self.imu_poll = self.parent().ui.dockWidget_sensor_manager.imu_poll
        self.trigger_poll = self.parent().ui.dockWidget_sensor_manager.trigger_poll
        self.trigger_sensor_handler = TriggerSensorHandler(self)
        self.engine = self.parent().ui.dockWidget_sensor_manager.engine
        self.last_imu_pos = np.zeros(3)

        self.quat_init = np.array([1.0, 0.0, 0.0, 0.0])
        self.timerId = None

        ## Test parameters
        self.test_handler = TestHandler(self)  # A container class to handle tests

        # Replay
        self.replay_handler = ReplayHandler(self)

    def init_Ui_Settings(self):
        """Initial UI layout, icons, and dynamic widgets (combo boxes, graph)."""
        self.setStyleSheet("QDockWidget::title" "{" "background : darkgray;" "}")
        self.loading_icons()

        # Settings
        #
        self.ui.spinBox_Ns.setReadOnly(True)
        self.ui.lineEdit_test_id.setReadOnly(True)
        self.ui.doubleSpinBox_dw.setReadOnly(True)

        # add CheckableComboBox at horizontalLayout_19
        self.ui.checkableCombox_test_ids = CheckableComboBox()
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ui.checkableCombox_test_ids.sizePolicy().hasHeightForWidth()
        )
        self.ui.checkableCombox_test_ids.setSizePolicy(sizePolicy)
        self.ui.horizontalLayout_19.insertWidget(3, self.ui.checkableCombox_test_ids)

        self.ui.comboBox_current_test = CustomComboBox(self.ui.groupBox_2)
        self.ui.comboBox_current_test.setObjectName("comboBox_current_test")
        self.ui.comboBox_current_test.setMinimumContentsLength(30)
        self.ui.horizontalLayout_13.insertWidget(1, self.ui.comboBox_current_test)

    def update_Ui_Settings(self):
        """Refresh dependent UI components after scene creation."""
        # self.ui.lineEdit_test_name.setText(self.test_id)
        self.load_plot_graph()

    def load_plot_graph(self):
        """Create the plot widget for the analysis tab and lines container."""
        self.plot_graph = Plot_Angles(
            holder_widget=self.ui.horizontalLayout_11, imu_poll=self.imu_poll
        )
        self.plot_graph.create_lines()

    def loading_icons(self):
        """Load icons for various push buttons (play, record, lock, refresh)."""
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(f"{ICONS_FOLDER}/play.png"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.Off,
        )
        icon.addPixmap(
            QtGui.QPixmap(f"{ICONS_FOLDER}/pause.png"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.On,
        )
        self.ui.pushButton_play_pause.setIcon(icon)

        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap(f"{ICONS_FOLDER}/record.png"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.Off,
        )
        icon1.addPixmap(
            QtGui.QPixmap(f"{ICONS_FOLDER}/record_g.png"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.On,
        )
        self.ui.pushButton_record.setIcon(icon1)

        icon2 = QtGui.QIcon()
        icon2.addPixmap(
            QtGui.QPixmap(f"{ICONS_FOLDER}/lock_open.png"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.Off,
        )
        icon2.addPixmap(
            QtGui.QPixmap(f"{ICONS_FOLDER}/lock_close.png"),
            QtGui.QIcon.Mode.Normal,
            QtGui.QIcon.State.On,
        )
        self.ui.pushButton_lock.setIcon(icon2)

        icon3 = QtGui.QIcon()
        icon3.addPixmap(
            QtGui.QPixmap(f"{ICONS_FOLDER}/refresh.png"),
            QtGui.QIcon.Mode.Normal,
        )
        self.ui.pushButton_replay_refresh.setIcon(icon3)
        self.ui.pushButton_refresh.setIcon(icon3)

    def connections(self):
        """Wire up all UI controls to their respective handlers."""
        # Visualization
        self.ui.pushButton_refresh.clicked.connect(self.update_imu_actor)
        self.ui.pushButton_plot.clicked.connect(self.plot_stream)
        self.ui.pushButton_play_pause.clicked.connect(self.play_pause_record_event)
        self.ui.pushButton_record.clicked.connect(self.start_stop_record_event)

        self.ui.pushButton_set_init_imus_orientations.clicked.connect(
            self.set_init_imus_orientations
        )
        self.ui.checkBox_compensate.stateChanged.connect(self.ctrl_imus_compensation)

        # Settings
        ## Test settings
        self.ui.pushButton_compute_parameters.clicked.connect(
            self.test_handler.compute_test_parameters
        )

        ## Test parameters

        self.ui.comboBox_current_test.currentTextChanged.connect(
            lambda: self.ui.doubleSpinBox_dw.setValue(self.test_handler.dw)
        )
        self.ui.pushButton_valid_test_params.clicked.connect(
            lambda: self.test_handler.test_validation_act()
        )

        ## Test ctrl
        self.ui.pushButton_ctrl_test.clicked.connect(self.test_handler.test_ctrl)

        # Sensors settings
        self.ui.pushButton_reset_all_triggers.clicked.connect(
            lambda: self.trigger_sensor_handler.reset_trigger_sensors()
        )

        # Replay
        self.ui.comboBox_replay_test_name.activated.connect(
            self.replay_handler.run
        )  # TODO: RM
        self.ui.comboBox_replay_test_name.hide()  # TODO: RM
        self.ui.checkableCombox_test_ids.model().dataChanged.connect(
            lambda: self.logger.info(f"{self.replay_handler.test_ids}")
        )
        self.ui.pushButton_plot_replay.clicked.connect(self.replay_handler.run)
        self.ui.pushButton_plot_replay_clear.clicked.connect(
            lambda: self.replay_handler.loading_plot_graph(reset=True)
        )

        self.ui.pushButton_replay_refresh.clicked.connect(
            self.replay_handler.refresh_test_list
        )

        self.ui.pushButton_export_csv.clicked.connect(
            lambda: self.replay_handler.export_csv()
        )

    def plot_stream(self):
        """Create the 3D scene on first click and refresh UI settings."""
        if hasattr(self, "scene"):
            pass
        else:
            self.create_scene(frame_holder=self.ui.frame_3D_visu)
            self.update_Ui_Settings()

    def start_stop_record_event(self, checked):
        """Start or stop recording across all connected IMUs."""
        if checked:
            [
                imu.record_start(self.ui.lineEdit_test_name.text())
                for _, imu in self.imu_poll.items()
            ]
            self.ui.pushButton_play_pause.setChecked(True)
            self.logger.info("Start recording...")
        else:
            [imu.record_stop() for _, imu in self.imu_poll.items()]
            self.logger.info("Stop recording")
            self.ui.pushButton_play_pause.setChecked(False)
            self.engine.dispose()

    def play_pause_record_event(self):
        """Toggle pause state across all connected IMUs."""
        [imu.record_pause_unpause() for _, imu in self.imu_poll.items()]

    def create_scene(self, frame_holder, **kwargs):
        """Build the VTK scene, add actors, set camera, and start the timer."""
        self.scene = SceneVTKQT(self.ui.frame_3D_visu)
        self.scene.create_plane_source()
        self.scene.set_button_press_event(
            "LeftButtonPressEvent", self.left_button_press_event
        )
        self.scene.set_button_press_event(
            "RightButtonPressEvent", self.right_button_press_event
        )
        self.scene.set_button_press_event(
            "MiddleButtonPressEvent", self.middle_button_press_event
        )
        self.scene.set_button_press_event("TimerEvent", self.timer_event)

        # Actors
        ## Earth reference frame
        self.earth_axes = vtkAxesActor()
        self.earth_axes.SetTotalLength(3.5, 3.5, 3.5)
        self.earth_axes.AxisLabelsOff()

        earth_transform = vtkTransform()
        earth_transform.Translate(
            -self.scene.source_plane_size / 4, -self.scene.source_plane_size / 4, 0
        )
        self.earth_axes.SetUserTransform(earth_transform)
        self.scene.add_actor(actor_name="earth_axes", actor_obj=self.earth_axes)

        ## IMU reference frame
        self.create_imus_actors()

        self.scene.set_camera_orientation_postion(
            view_up=np.array(
                [-0.32723178016995064, -0.266138685331967, 0.9066915474496151]
            ),
            focal_point=np.array(
                [-0.6935704447928486, 0.406629454137888, -0.38795919115991573]
            ),
            position=np.array(
                [13.043375096774954, 10.071626737629064, 7.406747657392687]
            ),
        )

        self.timerId = self.scene.iren.CreateRepeatingTimer(
            100
        )  # duration in milliseconds
        self.scene.iren.Initialize()
        self.scene.iren.Start()

    def create_imus_actors(self):
        """Create one axes actor per IMU currently connected."""
        for i, (key, imu) in enumerate(self.imu_poll.items()):
            self.create_imu_actor(key)

    def create_imu_actor(self, key):
        """Add an axes actor to the scene labeled with the IMU name."""
        self.scene.add_axes_actor(key, self.last_imu_pos, ZAxisLabelText=key)
        self.logger.info(f"Add actor: {key}, at postion: {self.last_imu_pos}")
        self.last_imu_pos[1] += 1

    def update_imu_actor(self):
        """Ensure actors exist for all IMUs and prune those no longer present."""
        self.logger.info("Refreshing scene...")
        for key in self.imu_poll:
            if key not in self.scene.actors_pool.keys():
                self.create_imu_actor(key)

        actor_to_rm = []
        for key in self.scene.actors_pool:
            if key not in self.imu_poll.keys() and key.startswith("IMU"):
                actor_to_rm.append(key)
        for key in actor_to_rm:
            _, self.last_imu_pos = self.scene.get_actor_orientation_position(
                self.scene.get_actor(key),
            )
            self.logger.info(f"Removing actor: {key}, at {self.last_imu_pos}")
            self.scene.remove_actor(key)

    def left_button_press_event(self, obj, event):
        camera = self.scene.ren.GetActiveCamera()
        self.logger.info(f"Camera postion:{camera.GetPosition()}")
        self.logger.info(f"Camera view up:{camera.GetViewUp()}")
        self.logger.info(f"Camera focal:{camera.GetFocalPoint()}")
        pass

    def right_button_press_event(self, obj, event):
        pass

    def middle_button_press_event(self, obj, event):
        pass

    def timer_event(self, obj, event):
        if self.ui.pushButton_ctrl_test.isChecked():
            self.test_handler.manage_test()

        if self.ui.pushButton_plot.isChecked():
            self.update_imus_orientations()
            self.plot_angles()

    def update_imus_orientations(self):
        [
            self.update_imu_orientation(key)
            for key in self.imu_poll
            if key in self.scene.actors_pool.keys()
        ]

    def update_imu_orientation(self, imu_name):
        imu = self.get_imu(imu_name)
        q = imu.q
        W, (X, Y, Z) = q.angle_deg, q.axis
        msg = imu_name + ": q="
        msg += f"[{q.w:.3f} {q.x:.3f} {q.y:.3f} {q.z:.3f}]"
        msg += "\r\n"
        self.ui.label_quaternions_values.setText(msg)

        transform = self.scene.get_actor(imu_name).GetUserTransform()
        position = transform.GetPosition()
        transform.Identity()  # Resetting Axes current rotations
        transform.Translate(position)
        transform.RotateWXYZ(W, X, Y, Z)
        self.scene.vtkWidget.Render()

    def get_imu(self, imu_name):
        """
        return imu object
        """
        return self.imu_poll[imu_name]

    def set_init_imus_orientations(self):
        for key, imu in self.imu_poll.items():
            imu.set_offset_quaternion()
            msg = key + ":"
            msg += np.array2string(imu.q_offset._q)
            msg += "\r\n"
            self.ui.label_quaterion_offset.setText(msg)

    def ctrl_imus_compensation(self, state):
        [imu.set_compensate(state) for _, imu in self.imu_poll.items()]

    def plot_angles(self):
        self.plot_graph.update_lines()
        for key, imu in self.imu_poll.items():
            # data_container = self.plot_graph.lines[key]
            # theta, phi, psi = (
            #     (np.degrees(imu.pitch)),
            #     np.degrees(imu.roll),
            #     np.degrees(imu.yaw),
            # )
            # data_container.update(psi)
            msg = f"\u03B8=[\u03B8\u2093:{imu.yaw:.3g}° \u03B8\u1D67:{imu.pitch:.3g}° \u03A8\u1D63:{imu.roll:.3g}°]"
            msg += "\r\n"
            self.ui.label_angles_values.setText(msg)

    def closeEvent(self, event):
        self.ui.pushButton_plot.setChecked(False)
        if self.timerId is not None:
            out = self.scene.iren.DestroyTimer(self.timerId)
            self.logger.info(f"Timer Event Closed: {out}")

    def close(self):
        self.ui.pushButton_plot.setChecked(False)
        if self.timerId is not None:
            out = self.scene.iren.DestroyTimer(self.timerId)
            self.logger.info(f"Timer Event Closed: {out}")


class DataContainer:
    """Container for a single plot line with rolling buffers."""

    line: object
    x: np.ndarray
    y: np.ndarray

    def __init__(self, line, y):
        self.line = line
        self.x = np.linspace(-1000, 0, 1000)
        self.y = y

    def update_x_y(self, y_new):
        """Shift buffers left and append the newest y value."""

        def _refresh_value(tab, new_val):
            tab[:-1] = tab[1:]
            tab[-1] = new_val

        # _refresh_value(self.x, x_new)
        # _refresh_value(self.y, y_new)
        self.y = np.roll(self.y, -1)
        self.y[-1] = y_new

        self.line.setData(self.y)

    def update_line(self):
        """Push current x/y arrays to the underlying plot line."""
        self.line.setData(self.x, self.y)

    def update(self, y_new):
        """Update buffers then redraw the line."""
        self.update_x_y(y_new)
        self.update_line()


class ImuDataPlotContainer:
    def __init__(self, Ns=1000):
        self.t = np.linspace(-Ns, 0, Ns + 1)
        self.yaw = np.linspace(-Ns, 0, Ns + 1)
        self.pitch = np.linspace(-Ns, 0, Ns + 1)
        self.roll = np.linspace(-Ns, 0, Ns + 1)
        self.q_w = np.linspace(-Ns, 0, Ns + 1)
        self.q_x = np.linspace(-Ns, 0, Ns + 1)
        self.q_y = np.linspace(-Ns, 0, Ns + 1)
        self.q_z = np.linspace(-Ns, 0, Ns + 1)

    def update(self, quat_new: Quaternion):
        self.t = np.roll(self.t, -1)
        self.yaw = np.roll(self.yaw, -1)
        self.pitch = np.roll(self.pitch, -1)
        self.roll = np.roll(self.roll, -1)
        self.q_w = np.roll(self.q_w, -1)
        self.q_x = np.roll(self.q_x, -1)
        self.q_y = np.roll(self.q_y, -1)
        self.q_z = np.roll(self.q_z, -1)

        # update data with new value
        # self.t[-1] = datetime.now(PARIS_TIME_ZONE).time()
        self.yaw[-1] = np.degrees(quat_new.yaw)
        self.pitch[-1] = np.degrees(quat_new.pitch)
        self.roll[-1] = np.degrees(quat_new.roll)
        # update data with new value
        self.q_w[-1] = quat_new.w
        self.q_x[-1] = quat_new.x
        self.q_y[-1] = quat_new.y
        self.q_z[-1] = quat_new.z


class Plot_Angles(pg.PlotWidget):
    """PlotWidget wrapper to manage multiple IMU plots and legends."""

    def __init__(self, holder_widget, imu_poll, parent=None, *args, **kargs):
        super().__init__(parent, *args, **kargs)
        self.lines = {}
        self.imu_poll = imu_poll
        self.Ns = 1000
        self.x = np.linspace(-self.Ns, 0, self.Ns + 1)
        self.variables = [
            "yaw",
            "pitch",
            "roll",
            # "q_w", "q_x", "q_y", "q_z"
        ]
        self.holder_widget = holder_widget
        self.angle_type = ["yaw", "pitch", "roll"]
        self.angle_unicode_symbol = ["\u03C8", "\u03B8", "\u03C6"]  # [phi, theta, psi]
        self.pen_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.styles = [
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenStyle.DashLine,
            QtCore.Qt.PenStyle.DotLine,
            QtCore.Qt.PenStyle.DashDotLine,
            QtCore.Qt.PenStyle.DashDotDotLine,
            QtCore.Qt.PenStyle.CustomDashLine,
        ]

        self.default_setup_graph()
        self.create_imu_container()
        # self.create_lines()

    def default_setup_graph(self):
        """Set plot title, styles, ranges and add the widget into its holder."""
        self.setTitle("Angles vs Time")
        self.setBackground("w")
        styles = {"color": "black", "font-size": "18px"}
        self.setLabel("left", "Angles (°)", **styles)
        self.setLabel("bottom", "Time (s)", **styles)
        self.showGrid(x=True, y=True)
        self.setYRange(-180, 180)
        self.setXRange(-self.Ns, 0)
        self.holder_widget.addWidget(self)

    def create_imu_container(self):
        """Allocate per-IMU rolling arrays for angles and quaternions."""
        self.containers = {}
        for key, imu in self.imu_poll.items():
            self.containers[key] = ImuDataPlotContainer(self.Ns)

    def create_lines(self):
        """Create yaw/pitch/roll plot lines for each connected IMU."""
        # Only applicable in real time analysis
        # TODO: move to right place
        for i, (key, imu) in enumerate(self.imu_poll.items()):
            self.lines[key] = {}
            for ii, var in enumerate(self.variables):
                self.addLegend()
                self.lines[key][var] = self.plot(
                    np.arange(self.Ns),
                    name=f"{key}_{var}",
                    pen=pg.mkPen(
                        color=self.pen_colors[i], width=2.5, style=self.styles[ii]
                    ),
                )

    def update_lines(self):
        for key, imu in self.imu_poll.items():
            if key in self.containers.keys():
                self.containers[key].update(quat_new=imu.q)
                for var in self.variables:
                    # x = getattr(self.containers[key], var)
                    y = getattr(self.containers[key], var)
                    self.lines[key][var].setData(self.x, y)
            else:
                for key in self.containers.keys():
                    for var in self.variables:
                        self.lines[key][var].clear()
                self.create_imu_container()
                self.create_lines()

    def add_line(
        self,
        key: str,
        linename: str = "",
        pen_color: tuple = (255, 0, 0),
        size=1000,
        *args,
        **kwargs,
    ):
        """
        Add a line for an IMU key, returning a DataContainer.

        Possible kwargs:
            symbol="+",
            symbolSize=15,
            symbolBrush="b",

        """
        pen = pg.mkPen(color=pen_color)

        line = self.plot(np.ones(size), name=linename, pen=pen, **kwargs)
        data_container = DataContainer(line, np.arange(size))
        self.lines.update({key: data_container})

    def update_plot(self, data_new: dict):
        """Update all managed lines from a dict of DataContainer objects."""

        for key, data in self.lines.items():
            data_new[key].update(data_new[key].x, data_new[key].y)

    def add_lines_from_imu_poll(self, imu_poll):
        for ind, (key, value) in enumerate(imu_poll.items()):
            self.add_line(key=key, linename=f"{key}", pen_color=self.pen_colors[ind])


class TriggerSensorHandler(QtCore.QObject):
    """Manage trigger devices, reset, and assign roles (starter/checker/closer)."""

    def __init__(self, parent):
        super(TriggerSensorHandler, self).__init__(parent)
        self.trigger_poll = self.parent().trigger_poll
        self.setup_logger()

        self.starter = None
        self.closer = None
        self.checkers = []

    def setup_logger(self):
        """Configure logger and attach to parent’s console handler when available."""
        self.logger = logging.getLogger(f"{TriggerSensorHandler.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        handler.setFormatter(LOG_DEFAULT_FORMATTER)
        self.logger.addHandler(handler)
        if hasattr(self.parent(), "handler"):
            self.handler = self.parent().handler
            self.handler.setFormatter(LOG_DEFAULT_FORMATTER)
            self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    def reset_trigger_sensors(self):
        """Send reset to all trigger devices."""
        self.logger.info("Resetting trigger sensors")
        for key, trigger in self.trigger_poll.items():
            trigger.reset()

    def triggers_roles_attribution(self):
        """Assign roles based on device labels and validate completeness."""
        # Define expected roles
        expected_roles = ["starter", "checker", "closer"]
        assigned_roles = []

        # Iterate through triggers and assign roles
        for key, trigger in self.parent().trigger_poll.items():
            role = trigger.label
            if role in expected_roles:
                if role == "checker":
                    self.checkers.append(trigger)
                else:
                    setattr(self, role, trigger)
                self.logger.info(f"Assigning role '{role}' to {key}")
                assigned_roles.append(role)
            else:
                raise ValueError(f"Trigger {trigger.name} has an unknown role: {role}")

        # Check if all roles are assigned
        missing_roles = set(expected_roles) - set(assigned_roles)
        if missing_roles:
            # Open a popup message asking to connect devices first
            msg = "Please connect all devices before proceeding."
            error_popup(title="Error", msg=msg, parent=self.parent())
            self.logger.error(f"Missing roles: {', '.join(missing_roles)}")
            self.parent().ui.pushButton_compute_parameters.setChecked(False)
            raise ValueError(f"Missing roles: {', '.join(missing_roles)}")


class TestHandler(QtCore.QObject):

    """
    Handle test parameter computation, validation, and execution control.

    Workflow
    --------
    1) Compute Parameters: build test plan and update UI.
    2) Validate: re-arm triggers, set test ID, confirm overwrites.
    3) Control: listen to triggers, record and process data streams.
    """

    def __init__(self, parent):
        super(TestHandler, self).__init__(parent)

        # Initialize parameters and logger
        self.engine = self.parent().engine
        self.R: float = None
        self.tests_container: Tests = None  # Needs to be compute after tests settings were set @see compute_test_parameters
        self._run_test_flag = False  # flag to stop the test TODO: use a QEvent instead?
        self.setup_logger()

    def setup_logger(self):
        """Configure logger and attach to parent’s console handler when available."""
        self.logger = logging.getLogger(f"{TestHandler.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        formatter = logging.Formatter(
            "[%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s()]: [%(name)s - \033[35m%(message)s\033[0m]"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if hasattr(self.parent(), "handler"):
            self.parent().handler.setFormatter(formatter)
            self.logger.addHandler(self.parent().handler)

        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    @property
    def test_ids(self):
        """List of generated test IDs."""
        return self.tests_container.test_names

    @property
    def test_id(self):
        """Currently selected test ID string from the UI."""
        return self.parent().ui.comboBox_current_test.currentText()

    @property
    def move_id(self):
        """Current patient/move identifier string from the UI."""
        return f"{self.parent().ui.lineEdit_move_id.text()}"

    @property
    def sw(self):
        """Shoulder width value from the UI."""
        return self.parent().ui.doubleSpinBox_shoulder_width.value()

    @property
    def visit(self):
        """Visit index (int) from the UI."""
        return int(self.parent().ui.comboBox_visit.currentText())

    @property
    def current_test_name(self):
        """Current test name from the UI combo box."""
        return self.parent().ui.comboBox_current_test.currentText()

    @property
    def current_test(self):
        """Return the Test instance corresponding to the current selection."""
        try:
            return self.tests_container[f"{self.current_test_name}"]
        except KeyError as e:  # no test selected
            raise ValueError("No current test selected") from e

    @property
    def current_ration(self):
        """Current ratio value for the selected test, or None."""
        try:
            return self.current_test.ratio
        except ValueError:  # no ratio selected
            return None  # TODO: raise error instead?

    @property
    def current_run(self):
        """Current run index for the selected test."""
        return self.current_test.run

    @property
    def dw(self):
        """Computed door width (ratio * shoulder width) or 0.0 if unavailable."""
        try:
            return self.current_ration * self.sw
        except TypeError:  # no ratio selected
            return 0.0  # TODO: raise error instead?

    def compute_test_parameters(self):
        """Insert patient in DB, assign trigger roles, and generate Tests list."""
        # Add current patients to DB
        patients = Patients(engine=self.engine, log_handler=self.parent().handler)
        patients.add_patient(move_id=self.move_id, shoulder_width=self.sw)
        self.triggers_roles_attribution()
        # Spawning tests
        self.tests_container = Tests(
            start_ratio=self.parent().ui.doubleSpinBox_ration_min.value(),
            end_ratio=self.parent().ui.doubleSpinBox_ration_max.value(),
            step=self.parent().ui.doubleSpinBox_step_size.value(),
            test_prefix_name=self.parent().ui.lineEdit_move_id.text(),
            sw=self.sw,
            visit=self.visit,
            engine=self.engine,
            additionnal_handler=self.parent().handler,
        )

        self.update_test_parameters_ui()

    def update_test_parameters_ui(self):
        """Refresh test lists in UI and enable validation when applicable."""
        self.parent().ui.comboBox_current_test.clear()
        # self.parent().ui.comboBox_current_test.addItems(self.test_ids)
        self.parent().ui.comboBox_current_test.addItemsFromDict(
            self.tests_container.test_dict
        )
        if self.dw != 0.0:
            self.parent().ui.doubleSpinBox_dw.setValue(self.dw)
            self.parent().ui.pushButton_valid_test_params.setDisabled(False)
        elif self.dw == 0.0:
            self.parent().ui.pushButton_valid_test_params.setDisabled(True)

    def rearm_trigger_sensors(self):
        """Reset trigger devices and wait until starter is not triggered."""
        self.parent().trigger_sensor_handler.reset_trigger_sensors()
        self.logger.info("Resetting trigger sensors...")
        while self.starter.state:
            self.logger.info("...")
        self.logger.info("Trigger sensors ready !")

    def test_validation_act(self):
        """Validate current test parameters and confirm overwrite if needed."""
        self.rearm_trigger_sensors()

        # updating test id value
        self.parent().ui.lineEdit_test_id.setText(self.test_id)

        if (
            self.tests_container[self.test_id].check_for_existing_test()
            and self.parent().ui.pushButton_valid_test_params.isChecked()
        ):
            msg = "⚠️ A completed test with the same parameters already exists.\n"
            msg += f"Do you want to continue and overwrite it?"
            resp = show_popup(title="Confirmation", msg=msg)

            if resp == False:
                self.parent().ui.pushButton_valid_test_params.setChecked(False)
                return False
            else:
                post_process = pp.PostProcessing.init_from_engine(self.engine)
                post_process.logger.addHandler(self.parent().handler)
                post_process.delete_entries({"test_id": self.test_id})
                self.tests_container[self.test_id].record_start()
                self.tests_container[self.test_id].status = TestStatus.TODO

        return True

    def test_ctrl(self):
        if self.parent().ui.pushButton_ctrl_test.isChecked():
            self.rearm_trigger_sensors()
            self.parent().logger.info(
                f"Starting test test_id={self.test_id}, dw={self.dw}"
            )
            self.parent().ui.lineEdit_test_name.setText(self.test_id)

            self.parent().ui.tabWidget.setCurrentIndex(1)
            self.parent().ui.pushButton_plot.setChecked(True)
            self.parent().plot_stream()

        else:
            self.stop_test()

    def triggers_roles_attribution(self):
        self.parent().trigger_sensor_handler.triggers_roles_attribution()
        self.starter = self.parent().trigger_sensor_handler.starter
        self.closer = self.parent().trigger_sensor_handler.closer
        self.checkers = self.parent().trigger_sensor_handler.checkers

    @execute_once
    def running_analysis(self):
        # 1 Recording imus data # DONE
        # 2 Recording for door trigger sensor # DONE
        # 3 Listening for ending analysis # DONE
        self.logger.info("Starter sensor have been triggered")
        self.logger.info("----------------- STARTING ANALYSIS -----------------")
        self.current_test.set_start_time()
        # handling imu recording
        [
            imu.record_start(test_id=self.test_id)
            for _, imu in self.parent().imu_poll.items()
        ]

        # handling checker trigger sensor
        [checker.log_triggering(test_id=self.test_id) for checker in self.checkers]

    def manage_test(self):  # impletementation witout loop
        """
        This function is called repeatly by timer event from 3D plot scene
        """
        if self.starter.state:
            self.running_analysis()

        if self.closer.state:
            self.stop_test()
            self.parent().ui.pushButton_valid_test_params.setChecked(False)
            self.parent().ui.pushButton_ctrl_test.setChecked(False)
            success = self.check_post_test()
            if success:
                # raise tab_replay_record
                self.tests_container[self.test_id].status = TestStatus.PASSED
                self.parent().replay_handler.refresh_test_list()
                self.parent().ui.tabWidget.setCurrentIndex(2)
                self.parent().ui.checkableCombox_test_ids.checkItemFromName(
                    self.test_id
                )
                self.parent().replay_handler.run()
            else:
                self.tests_container[self.test_id].status = TestStatus.FAILED
                self.parent().ui.tabWidget.setCurrentIndex(0)
                self.parent().ui.pushButton_valid_test_params.setChecked(False)

            self.parent().ui.comboBox_current_test.update_lines_styles()

    def stop_test(self):
        self._run_test_flag = False
        self.running_analysis.reset()  # Reset the decorator state to allow re-execution
        self.parent().ui.pushButton_ctrl_test.setChecked(False)
        [
            imu.record_stop()
            for _, imu in self.parent().imu_poll.items()
            if imu.is_recording
        ]
        self.current_test.set_end_time()  # Set the end time of the test in the database
        self.logger.info(" ----------------- STOP ANALYSIS  ----------------- ")

    def check_post_test(self):
        self.logger.info("Checking post test...")
        for name, imu in self.parent().imu_poll.items():
            imu_post_process = pp.IMUObject.init_from_engine(
                engine=self.engine,
                tablename=name,
                filter_dict={"test_id": self.test_id},
            )
            success = imu_post_process.was_imu_disconnected()
            if success:
                # Show a pop windows that says that test was a failure
                self.logger.info("Test failed")
                msg = (
                    f"⚠️ A time gap superior to 2s was detected from "
                    + f"{name} "
                    + "device.\n"
                )
                msg = "The current test may be not reliable.\n"
                msg += f"We recommend to redo your analysis"
                resp = error_popup(title="Error detected", msg=msg)
                return False
        return True  # If no errors were found, return True

    def get_all_test_ids(self):
        try:
            return db.get_attr_entries_from_model("test_id", self.session, self.model)
        except Exception as e:
            self.logger.error(f"An error has arrived, see error log: {e}")
            return []


class ReplayHandler(QtCore.QObject):
    def __init__(self, parent):
        super(ReplayHandler, self).__init__(parent)

        self.setup_logger()

        self.engine = self.parent().engine

        self.color_index = 0
        self.imu_poll = {}
        self.trigger_poll = {}
        self.imu_data = {}
        self.checked_time = None

        self.record_setting()

        self.refresh_test_list()

    def create_imu_poll(self):
        metadata = MetaData()
        metadata.reflect(bind=self.engine)
        for table in metadata.sorted_tables:
            self.logger.info(f"IMU Table name: {table.name}")
            if table.name.startswith("IMU"):
                imu = IMU.init_from_db_engine(name=table.name, engine=self.engine)
                self.imu_poll.update({table.name: imu})

    def create_polls(self):
        # Create imu poll not from connected devices but from device existing in DB
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        # Affichez les noms des tables
        for table in metadata.sorted_tables:
            if table.name.startswith("IMU"):
                imu = IMU.init_from_db_engine(name=table.name, engine=self.engine)
                self.imu_poll.update({table.name: imu})
            elif table.name.startswith("TRIGGER"):
                trigger = Trigger.init_from_db_engine(
                    name=table.name, engine=self.engine
                )
                self.trigger_poll.update({table.name: trigger})

    @property
    def test_id(self) -> str:  # TODO: RM
        return self.parent().ui.comboBox_replay_test_name.currentText()

    @property
    def test_ids(self) -> list:
        return self.parent().ui.checkableCombox_test_ids.get_selected_items()

    @property
    def standalone_test(self) -> bool:
        return self.parent().ui.checkBox_standalone.isChecked()

    def setup_logger(self):
        self.logger = logging.getLogger(f"{ReplayHandler.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        formatter = logging.Formatter(
            "[%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s()]: [%(name)s - \033[33m%(message)s\033[0m]"
        )
        handler.setFormatter(formatter)
        if hasattr(self.parent(), "handler"):
            self.parent().handler.setFormatter(formatter)
            self.logger.addHandler(self.parent().handler)

        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    def refresh_test_list(self):
        self.loading_plot_graph(reset=True)
        self.parent().ui.comboBox_replay_test_name.clear()  # TODO RM
        self.parent().ui.checkableCombox_test_ids.clear()

        # check for test_id only in imus tables
        test_ids = sorted(self.get_imus_test_ids())
        self.parent().ui.comboBox_replay_test_name.addItems(test_ids)  # TODO RM
        self.parent().ui.checkableCombox_test_ids.addItems(test_ids)

        self.logger.info(f"{self.test_ids}")

    def get_imus_test_ids(self):
        self.create_imu_poll()
        test_ids = []
        try:
            for name, imu in self.imu_poll.items():
                test_ids += imu.get_test_ids()
        except Exception as e:
            self.logger.error(f"An error has arrived, see error log: {e}")

        return list(set(test_ids))

    def check_for_imus_data(self, test_id):
        imu_check = 0
        imu_key_to_delete = []
        for name, imu in self.imu_poll.items():
            imu_check += len(imu.get_entries_by_test_id(test_id))
            if len(imu.get_entries_by_test_id(test_id)) == 0:
                # delete key from self.imu_poll
                imu_key_to_delete.append(name)
        for key in imu_key_to_delete:
            del self.imu_poll[key]

        if imu_check == 0:
            self.logger.info(f"No entrie in IMU tables - {test_id}")
            return False
        else:
            self.logger.info(f"✅ Imu entries detectes - {test_id}")
            return True

    def check_for_test_data(self, test_id):
        # Check in IMU tables
        self.check_for_imus_data(test_id)
        # Check in trigger tables
        trig_check = 0
        for _, trig in self.trigger_poll.items():
            trig_check += len(trig.get_entries_by_test_id(test_id))

        if trig_check == 0:
            self.logger.info("No entries in trigger table")
            return False
        else:
            self.logger.info("✅ Triggers entries detected")

        return True

    def loading_plot_graph(self, reset=True):
        """Create or reset the analysis plot widget for replay mode."""
        if not hasattr(self, "plot_graph"):
            self.color_index = 0
            self.plot_graph = Plot_Angles(
                holder_widget=self.parent().ui.horizontalLayout_18,
                imu_poll=self.imu_poll,
                axisItems={"bottom": TimeAxisItem(orientation="bottom")},
            )
        else:
            if reset:
                self.plot_graph.clear()
                self.color_index = 0
            pass

    def list_imu_variables(self):
        """Return the list of IMU variable names used for plotting."""
        return ["q_Z", "q_X ", "q_Y", "q_W", "yaw", "pitch", "roll"]

    def get_imu_data(self, test_id):
        """Load post-processed IMU objects for a test ID using the engine."""
        for name, imu in self.imu_poll.items():
            post_processing = pp.IMUObject.init_from_engine(
                self.engine, tablename=f"{name}", filter_dict={"test_id": test_id}
            )
            self.imu_data[name] = post_processing

    def get_trigger_data(self, test_id):
        """Load trigger timestamps per device for a test ID using the engine."""
        self.trigger_data = {}
        # Getting checked time
        for name, trigger in self.trigger_poll.items():
            checked_time = trigger.get_time_from_test_id(test_id)
            if len(checked_time) > 0:
                self.trigger_data[name] = checked_time
        return self.trigger_data

    @staticmethod
    def time_to_seconds(t):
        """Convert a datetime.time to seconds (int)."""
        return t.hour * 3600 + t.minute * 60 + t.second

    @staticmethod
    def time_to_milliseconds(t):
        """Convert a datetime.time to milliseconds (int)."""
        return (t.hour * 3600 + t.minute * 60 + t.second) * 1000 + t.microsecond // 1000

    def get_new_color(self, inc=1):
        """Return the next distinct plot color using PyQtGraph’s palette."""
        self.color_index += inc
        c = pg.intColor(self.color_index)
        return c

    def plot(self, test_id, reset=True):
        """Plot selected angles and triggers for a given test ID."""
        self.loading_plot_graph(reset)
        self.init_time = None
        for ind, (key, imu_data) in enumerate(self.imu_data.items()):
            Ns = len(imu_data.times)
            self.parent().ui.spinBox_qoffset_index.setMinimum(-1)
            self.parent().ui.spinBox_qoffset_index.setMaximum(Ns)
            chosen_angle = self.parent().ui.comboBox_chose_angle.currentText()

            if Ns > 0:
                self.plot_graph.addLegend()
                # data_container = self.plot_graph.lines[key + "_\u03A8"]
                # data_container.set_batch_data(imu_data["time"], imu_data["yaw"])
                x = [self.time_to_milliseconds(t) for t in imu_data.times]
                if self.init_time is None:
                    self.init_time = x[0]
                x = [xx - self.init_time for xx in x]
                # pen = pg.mkPen(color=self.plot_graph.pen_colors[ind])
                pen = pg.mkPen(color=self.get_new_color(), width=2.5)
                qoffset_index = self.parent().ui.spinBox_qoffset_index.value()
                if chosen_angle != "norm_acc":
                    self.plot_graph.plot(
                        x=x,
                        y=imu_data.apply_compensation(qoffset_index)
                        .get_angle_df(f"{chosen_angle}")
                        .values.astype(np.float64),
                        name=test_id + "_" + key + "_" + f"{chosen_angle}",
                        pen=pen,
                        # symbol ='o',
                        # symbolSize=5
                    )
                elif chosen_angle == "norm_acc":
                    self.plot_graph.plot(
                        x=x,
                        y=imu_data.norm_acc,
                        name=test_id + "_" + key + "_" + f"{chosen_angle}",
                        pen=pen,
                    )

        if self.check_for_test_data(test_id):
            for name, checked_time in self.trigger_data.items():
                checked_time_ms = [self.time_to_milliseconds(t) for t in checked_time][
                    0
                ]
                checked_time_ms = checked_time_ms - self.init_time
                pen = pg.mkPen(
                    color=self.get_new_color(),
                    width=2,
                    style=QtCore.Qt.PenStyle.DotLine,
                )
                self.plot_graph.plot(
                    x=np.array([checked_time_ms, checked_time_ms]),
                    y=np.array([-180, 180]),
                    name=test_id + "_" + name,
                    pen=pen,
                )

        vb = self.plot_graph.getViewBox()
        vb.setRange(yRange=(-180, 180))
        vb.enableAutoRange(axis="x", enable=True)

        self.color_index += 3

    def run(self):
        self.create_polls()
        for test_id in self.test_ids:
            self.get_imu_data(test_id)
            self.get_trigger_data(test_id)
            self.plot(test_id, reset=False)

    def record_setting(self, engine=None):
        if not hasattr(self, "engine"):
            self.engine = engine
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        return self

    def export_csv(self):
        # TODO : enhance this
        exporter = pg.exporters.CSVExporter(self.plot_graph.getPlotItem())
        exporter.export()


class TimeAxisItem(pg.AxisItem):
    """AxisItem to format x-axis ticks as HH:MM:SS.mmm timestamps."""

    def tickStrings(self, values, scale, spacing):
        # Convert each value (milliseconds) to an HH:MM:SS.mmm string
        result = []
        for value in values:
            milliseconds = int(value % 1000)
            seconds = int((value // 1000) % 60)
            minutes = int((value // 60000) % 60)
            hours = int(value // 3600000)
            result.append(f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")
        return result


if __name__ == "__main__":
    import sys
    from PyQt6.QtCore import Qt
    from record.gui._debug import Debug

    app = QtWidgets.QApplication(sys.argv)
    debug = Debug()
    # debug.connect_to_imu(name="IMU_BNO055_00", ip_address="192.168.0.121")
    # debug.connect_to_trigger(
    #     name="TRIGGER_SENSOR_00", ip_address="192.168.0.21", port=64388, label="starter"
    # )
    # debug.connect_to_trigger(
    #     name="TRIGGER_SENSOR_01", ip_address="192.168.0.135", port=64388, label="closer"
    # )
    # debug.connect_to_trigger(
    #     name="TRIGGER_SENSOR_02",
    #     ip_address="192.168.0.223",
    #     port=64388,
    #     label="checker",
    # )
    debug.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, SequenceDockWidget(debug))
    debug.show()
    debug.resize(800, 600)
    sys.exit(app.exec())
