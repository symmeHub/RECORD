from record.core.imu import IMU
from record.core.trigger import Trigger
from record.gui.sensor_manager import ConfigFile
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget
from sqlalchemy import create_engine
import logging


class Debug(QMainWindow):
    # engine = create_engine(f"sqlite:////workspaces/record/assets/database.db")
    engine = create_engine(f"sqlite:////Users/celmo/Git/record/assets/database.db")

    def __init__(self):
        super(Debug, self).__init__()
        self.create_virtual_attr()

        self.logger = logging.getLogger(f"_debug.py")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    @staticmethod
    def set_nested_attribute(obj, attr_path, value):
        # Divise le chemin des attributs en parties
        attrs = attr_path.split(".")

        # Navigue dans les attributs en créant les sous-attributs si nécessaire
        for attr in attrs[:-1]:  # Parcourt tout sauf le dernier attribut
            if not hasattr(obj, attr):
                setattr(
                    obj, attr, type("DynamicObject", (object,), {})()
                )  # Crée un nouvel objet dynamique
            obj = getattr(obj, attr)

        # Définit l'attribut final avec la valeur souhaitée
        setattr(obj, attrs[-1], value)

    def create_virtual_attr(self):
        self.create_imu_poll()
        self.create_trigger_poll()
        self.create_engine()

    def load_config(self, filename: str = "./../../assets/config.yml"):
        self.config = ConfigFile(self.filename_config)

    def create_imu_poll(self):
        self.set_nested_attribute(self, "ui.dockWidget_sensor_manager.imu_poll", {})

    def create_trigger_poll(self):
        self.set_nested_attribute(self, "ui.dockWidget_sensor_manager.trigger_poll", {})

    def create_engine(self):
        self.set_nested_attribute(
            self, "ui.dockWidget_sensor_manager.engine", self.engine
        )

    def connect_to_imu(
        self,
        name: str = "IMU_BNO055_00",
        ip_address: str = "192.168.0.27",
        port: int = 64344,
        label="imu_right",
    ):
        imu = IMU(name, ip_address=ip_address, port=port, label=label)
        imu.connect()
        imu.record_setting(self.engine)
        self.ui.dockWidget_sensor_manager.imu_poll.update({name: imu})

    def connect_to_trigger(
        self,
        name: str = "TRIGGER_SENSOR_00",
        ip_address: str = "192.168.0.21",
        port: int = 64388,
        label="starter",
    ):
        trigger = Trigger(name, ip_address=ip_address, port=port, label=label)
        trigger.connect()
        trigger.record_setting(self.engine)
        self.ui.dockWidget_sensor_manager.trigger_poll.update({name: trigger})

    def closeEvent(self, event):
        [dock.close() for dock in self.findChildren(QDockWidget)]


if __name__ == "__main__":
    import sys
    from PyQt6.QtCore import Qt
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)

    debug = Debug()
    debug.connect_to_imu()
    debug.connect_to_trigger()
    # debug.show()
    # debug.resize(800, 600)
    # sys.exit(app.exec())
"""
Developer helper window to spin up the Sequence dock quickly for testing.
"""
