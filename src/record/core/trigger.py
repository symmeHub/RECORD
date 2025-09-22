"""
Trigger device client to log external events (e.g., start/stop markers).
"""

from datetime import datetime
from time import sleep, time

from sqlalchemy.orm import sessionmaker
from record.core.device import Device
from record.core.constants import PARIS_TIME_ZONE
from record.core.decorators import threaded
from record.core.database import TriggerPacketDB
from colorama import Fore
from colorama import Style


import numpy as np


class Trigger(Device):
    """
    TCP-trigger device handling event notifications and database logging.
    """

    def __init__(self, name: str, ip_address: str, port: int, label: str = None):
        super().__init__(name=name, ip_address=ip_address, port=port, label=label)
        self.TIMEOUT = None
        self.attempt = 0

        self.data = None  # Must be define
        self.state = False
        self.info = None

        self.flag_record_data = False
        self.flag_record_pause = False

    def get_processing_func(self):
        """
        Return the socket processing function for trigger messages.
        """

        def processing_socket_data(obj):
            buffer = b""
            while True:
                chunk = obj.socket.recv(1024)
                if not chunk:
                    raise ConnectionError("Connection interrupted")
                buffer += chunk  # Ajout des données au tampon
                string_data = buffer.decode("utf-8")
                if "\n" in string_data:
                    if not "IDLE" in string_data:
                        obj.logger.info(f" Received msg: {string_data.strip()}")
                    break  # Si une nouvelle ligne est reçue, on arrête de recevoir des données
            if "TRIGGERED" in string_data:
                # obj.logger.info(f" Received msg: {string_data.strip()}")
                obj.socket.send("ACK_TRIGGERED\n".encode("utf-8"))
                obj.state = True
                sleep(1)
            elif "ACK_RESET" in string_data:
                # obj.logger.info(f" Received msg: {string_data.strip()}")
                obj.logger.info(f"Trigger sensor have been reset")
                obj.state = False
            elif "ACK_STATUS" in string_data:
                # obj.logger.info(f" Received msg: {string_data.strip()}")
                obj._sensor_status = string_data.split(" ")[-1]
            elif "ACK_INFO" in string_data:
                obj.info = string_data.replace("ACK_INFO", "")

        return processing_socket_data

    def send_request(self, request_key: str):
        """
        Send a request line to the trigger device.

        Allowed keys typically include: REQ_RESET, REQ_STATUS, REQ_INFO, REQ_AUTO_TRESH_MAX.
        """

        try:
            self.logger.info("Sending request to ESP32...")
            self.socket.sendall(
                f"{request_key}\n".encode()
            )  # Envoie de la commande "reset"
            sleep(1)  # Laisse un délai pour que l'ESP32 traite la commande
        except Exception as e:
            self.logger.error(f"Error while sending request: {e}")
            return False

    def send_reset_request(self):
        """Send a reset request to the ESP32 and wait for acknowledgment."""
        self.send_request("REQ_RESET")

    def send_status_request(self):
        """Request the device status and wait for acknowledgment."""
        self.send_request("REQ_STATUS")

    def send_info_request(self):
        """Request device info and wait for acknowledgment."""
        self.send_request("REQ_INFO")

    def send_auto_treshMax_request(self):
        """Ask device to auto-compute a threshold maximum value."""
        self.send_request("REQ_AUTO_TRESH_MAX")

    def set_tresh_max(self, tresh_max=1800):
        """Set device threshold maximum value."""
        try:
            self.logger.info("Sending tresh max value..")
            self.socket.sendall(
                f"TRESH_MAX#{tresh_max}\n".encode()
            )  # Envoie de la commande
        except Exception as e:
            self.logger.error(f"Error while sending tresh max: {e}")
            return False

    def set_tresh_min(self, tresh_min=1800):
        """Set device threshold minimum value."""
        try:
            self.logger.info("Sending tresh min value..")
            self.socket.sendall(
                f"TRESH_MIN#{tresh_min}\n".encode()
            )  # Envoie de la commande
        except Exception as e:
            self.logger.error(f"Error while sending tresh min: {e}")
            return False

    def sync(self):
        """
        Attempt to reconcile internal state with device status.
        """
        self.send_reset_request()
        sleep(1)
        if hasattr(self, "_sensor_status"):
            if self._sensor_status == "DONE" and self.state == False:
                self.send_reset_request()
            if self._sensor_status == "READY" and self.state == True:
                self.state = False

    def ping_data(self):
        """Fetch and log device information string."""
        self.send_info_request()
        sleep(0.5)
        self.logger.info(f"\033[33m{self.info}\033[0m")

    def is_data_valid(self) -> bool:
        """
        Always True for the trigger device.
        """
        return True

    @threaded
    def reset(self):
        self.state = False
        self.sync()

    def record_setting(self, engine):
        """
        Bind an engine and ensure the trigger table exists.
        """
        if not hasattr(self, "engine"):
            self.engine = engine
        Session = sessionmaker(bind=self.engine)
        session = Session()
        self.model = TriggerPacketDB.get_model(tablename=self.name)
        self.model.metadata.create_all(self.engine)
        session.close()

    @threaded
    def record_start(
        self,
        test_id: str,
        record_freq: float = 10,
        nb_commit_per_record_period: int = 100,
    ):
        """
        Start logging trigger state to the database periodically.
        """
        record_period = 1 / record_freq
        entries = []
        self.flag_record_data = True
        self.logger.info("\033[33mStart recording...\033[0m")
        t0 = time()
        while self.flag_record_data:
            if self.flag_record_pause:
                self.logger.info("Recording paused...")
                while self.flag_record_pause:
                    sleep(1)  # Attend la reprise
                self.logger.info("Resume recording.")
            elapsed_time = time() - t0
            if elapsed_time < record_period:
                sleep(record_period - elapsed_time)
            new_entry = self.model(
                time=datetime.now(PARIS_TIME_ZONE).time(),
                test_id=test_id,
                state=self.state,
            )
            entries.append(new_entry)
            if len(entries) >= nb_commit_per_record_period:
                self.save_entries(entries)
            t0 = time()
        if len(entries) > 0:
            self.save_entries(entries)  # Enre

    @threaded
    def log_triggering(self, test_id: str, record_freq: float = 20):
        """
        Log a single trigger event timestamp when `state` becomes True.
        """
        record_period = 1 / record_freq
        t0 = time()
        self.flag_record_data = True
        while self.flag_record_data:
            elapsed_time = time() - t0
            if elapsed_time < record_period:
                sleep(record_period - elapsed_time)
            if self.state:
                new_entry = self.model(
                    time=datetime.now(PARIS_TIME_ZONE).time(),
                    test_id=test_id,
                    state=self.state,
                )
                self.save_entries([new_entry])
                break
            t0 = time()

    def save_entries(self, entries):
        """
        Bulk-save a list of ORM entries using a short-lived session.
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            session.bulk_save_objects(entries)
            session.commit()
            entries.clear()
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            # session.rollback()
        finally:
            session.close()

    def get_time_from_test_id(self, test_id):
        """
        Retrieve a list of `datetime.time` entries for a given test.
        """
        entries = self.get_entries_by_test_id(test_id=test_id)
        Nr = len(entries)
        times = [None] * Nr
        for i, entry in enumerate(entries):
            times[i] = entry.time
        return times


if __name__ == "__main__":
    from sqlalchemy import create_engine
    import uuid

    engine = create_engine(f"sqlite://///Users/celmo/Git/record/assets/database.db")

    # Configuration du client IMU
    trigger = Trigger("TRIGGER_SENSOR_00", "192.168.0.21", 64388, "starter")
    trigger.connect()
    test_id = uuid.uuid4().int >> (128 - 32)
    trigger.record_setting(engine)
    # for _ in range(100):
    #     sleep(0.1)
    #     print(trigger.data)
