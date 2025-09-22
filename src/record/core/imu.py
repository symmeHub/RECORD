"""
IMU device client handling quaternion/accel stream ingestion and logging.

Provides TCP ingestion, live quaternion compensation, and background DB writer.
"""

import struct
import sys
from datetime import datetime
import logging
from time import sleep, time
import math

from sqlalchemy.orm import sessionmaker

from record.core.constants import PARIS_TIME_ZONE
from record.core.device import Device
from record.core.decorators import threaded
from record.core.database import IMUPacketDB
from record.core.geometry import Quaternion

from queue import Queue
from threading import Thread, Event


import numpy as np


class IMU(Device):
    def __init__(self, name: str, ip_address: str, port: int, label: str = None):
        """
        Initialize an IMU client that connects to a TCP server.

        Parameters
        ----------
        name : str
            IMU name (used for logging and DB tablename).
        ip_address : str
            Server IP address.
        port : int
            Server port.
        label : str, optional
            Optional human-readable label.
        """
        super().__init__(name=name, ip_address=ip_address, port=port, label=label)
        self.TIMEOUT = 1
        self.attempt = 0

        self.BUFFER_SIZE = 10
        self.N_data = 8  # qw, qx, qy, qz, t, ax, ay, az
        self.PACKET_SIZE = self.N_data * 4
        self.PACKET_FORM = f"{str(int(self.N_data))}f"
        self.data = np.zeros([self.BUFFER_SIZE, self.N_data])
        self.data[:, 0] = 1.0
        self.buffer = b""

        self.compensate = False
        self._q = Quaternion(np.array([1, 0, 0, 0]))
        self._q_offset = Quaternion(np.array([1, 0, 0, 0]))
        self._q_compensate = Quaternion(np.array([1, 0, 0, 0]))

        self._flag_listen_data = False

        self.flag_record_data = False
        self.flag_record_pause = False

        # Configuration du logger
        self.logger = logging.getLogger(f"{name}")
        self.setup_logger()

        self._roll = np.nan
        self._pitch = np.nan
        self._yaw = np.nan

        self.previous_elapse_time = 0.0

        self.db_queue = None
        self.db_thread = None
        self.db_stop_event = None

    @property
    def q(self) -> Quaternion:
        return self.get_quaternion()

    @property
    def q_offset(self) -> Quaternion:
        return self._q_offset

    @property
    def q_compensate(self) -> Quaternion:
        return self.get_compensate_quaternion()

    @property
    def timestamp(self) -> np.int32:
        return self.data[0, 4]

    @property
    def yaw(self):
        # Lacet (yaw)
        self._yaw = self.q.yaw
        return self._yaw

    @property
    def pitch(self):
        # Tangage (pitch)
        self._pitch = self.q.pitch
        return self._pitch

    @property
    def roll(self):
        # Roulis (roll)
        self._roll = self.q.roll
        return self._roll

    def is_data_valid(self) -> bool:
        """
        Return True if the quaternion is not null
        """
        if bool((self.q.w == 0) or (self.q.imag_part.sum() == 0)):
            raise ValueError("Quaternion is null")
        return True

    def set_compensate(self, state: bool):
        """Enable/disable quaternion offset compensation during reads."""
        self.compensate = state

    def ping_data(self):
        """Log a formatted snapshot of the current quaternion and timestamp."""
        self.logger.info(
            f"\nReceived Quat: w={self.q.w:.4f}, x={self.q.x:.4f}, y={self.q.y:.4f}, z={self.q.z:.4f}\ntimestamp={self.timestamp} [ms]"
        )
        # self.logger.info(f"{self.data[0, 0:-1]}")

    def get_processing_func(self):
        """
        Select an OS-specific socket data processing function.

        Returns a callable that continuously reads from the socket, unpacks
        packets and updates the rolling buffer.
        """

        def processing_socket_data(obj):
            try:
                raw_packet = obj.socket.recv(1024)  # Lire plus large
                if not raw_packet:
                    raise ConnectionError("Connection interrupted")
                obj.buffer += raw_packet

                # Traite tous les paquets complets disponibles
                while len(obj.buffer) >= obj.PACKET_SIZE:
                    packet_bytes = obj.buffer[: obj.PACKET_SIZE]
                    obj.buffer = obj.buffer[obj.PACKET_SIZE :]
                    packet = struct.unpack(obj.PACKET_FORM, packet_bytes)
                    obj.update_data(packet)
                    sleep(2e-3)

            except Exception as e:
                obj.logger.error(f"Socket error: {e}")
                raise

        def processing_socket_data_darwin(obj):
            try:
                raw_packet = obj.socket.recv(1024)  # Lire plus large
                if not raw_packet:
                    raise ConnectionError("Connection interrupted")
                obj.buffer += raw_packet

                # Traite tous les paquets complets disponibles
                while len(obj.buffer) >= obj.PACKET_SIZE:
                    packet_bytes = obj.buffer[: obj.PACKET_SIZE]
                    obj.buffer = obj.buffer[obj.PACKET_SIZE :]
                    packet = struct.unpack(obj.PACKET_FORM, packet_bytes)
                    obj.update_data(packet)
                    # sleep(2e-3)

            except Exception as e:
                self.logger.error(f"Socket error: {e}")
                raise

        proc_func_dic = {
            "darwin": processing_socket_data_darwin,
            "linux": processing_socket_data,
            "win32": processing_socket_data,
        }

        return proc_func_dic[sys.platform]

    def update_data(self, packet):
        """Update the rolling buffer with a newly received packet tuple."""
        if self.previous_elapse_time != packet[4]:
            self.previous_elapse_time = packet[4]
            self.data[:-1] = self.data[1:]

            for i in range(self.N_data):
                self.data[-1, i] = packet[i]

        # # quaternion qw qx  qy  qz
        # self.data[-1, 0] = packet[0]
        # self.data[-1, 1] = packet[1]
        # self.data[-1, 2] = packet[2]
        # self.data[-1, 3] = packet[3]

        # # elapse time
        # self.data[-1, 4] = packet[4]

        # # accel x y z
        # self.data[-1, 5] = packet[5]
        # self.data[-1, 6] = packet[6]
        # self.data[-1, 7] = packet[7]

    def get_quaternion(self):
        """Return the current Quaternion (compensated if `self.compensate` is True)."""
        self._q.set_q(self.data[0, :4])
        return self.q_compensate if self.compensate else self._q

    def get_acc(self):
        """Return the latest acceleration sample as (ax, ay, az)."""
        return self.data[-1, -4:]

    def record_setting(self, engine=None):
        """Bind an engine and ensure the IMU table exists for this device name."""
        if engine is not None:
            self.engine = engine
        else:
            if not hasattr(self, "engine"):
                raise AttributeError("Engine attribute is missing")

        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.model = IMUPacketDB.get_model(tablename=self.name)
        self.model.metadata.create_all(self.engine)

    @threaded
    def record_start(
        self,
        test_id: str,
        imu_elapsed_time_unit: str = "ms",
        record_freq: float = 30,
    ):
        """
        Starts recording IMU data in to sql-database at a specified frequency.

        Args:
            test_name (str): Name of the test.
            record_freq (float, optional): Frequency at which to record data in Hz. Defaults to 20.
            nb_commit_per_record_period (int, optional): Number of data packets to commit to the database per record period. Defaults to 10.
        """

        # record_period = 1 / record_freq

        self.db_queue = Queue()
        self.db_stop_event = Event()
        self.db_thread = Thread(
            target=database_worker,
            args=(
                self.engine,
                self.model,
                self.db_queue,
                self.db_stop_event,
                self.logger,
            ),
            daemon=True,
        )
        self.db_thread.start()

        self.flag_record_data = True
        self.logger.info("\033[33mStart recording...\033[0m")
        while self.flag_record_data:
            if self.flag_record_pause:
                self.logger.info("Recording paused...")
                while self.flag_record_pause:
                    sleep(1)  # Attend la reprise
                self.logger.info("Resume recording.")
            # t0 = time()
            # elapsed_time = time() - t0
            # if elapsed_time < record_period:
            #     sleep(record_period - elapsed_time)
            sleep(2e-2)
            new_entry = self.model(
                time=datetime.now(PARIS_TIME_ZONE).time(),
                test_id=test_id,
                q_w=self.data[0, 0],
                q_x=self.data[0, 1],
                q_y=self.data[0, 2],
                q_z=self.data[0, 3],
                imu_elapsed_time=self.data[0, 4],
                imu_elapsed_time_unit=imu_elapsed_time_unit,
                acc_x=self.data[0, 5],
                acc_y=self.data[0, 6],
                acc_z=self.data[0, 7],
            )
            self.db_queue.put(new_entry)

    def record_stop(self):
        """
        Stop data recording.

        This method stops the recording process and commits any remaining records to the database.
        It also logs a message indicating that the recording has stopped and the database has been disconnected.
        """
        self.flag_record_data = False
        self.flag_record_pause = False
        self.logger.info("\033[33mRecording Stopped\033[0m")
        if self.db_stop_event and self.db_thread:
            self.db_stop_event.set()
            self.db_thread.join()
        self.logger.info("Database Disconnected")
        sleep(0.5)

    def set_offset_quaternion(self):
        """Capture the current quaternion as the offset for compensation."""
        self._q_offset.set_q(self._q._q)

    def get_compensate_quaternion(self) -> Quaternion:
        """Return the compensation quaternion q = q_current^{-1} * q_offset."""
        q = self._q.q_inv * self._q_offset
        return q

    def get_quaternion_from_test_id(self, test_id):
        """Load all quaternions for a given test ID from the database."""
        entries = self.get_entries_by_test_id(test_id=test_id)
        Nr = len(entries)
        quaternions = [None] * Nr
        for i, entry in enumerate(entries):
            quaternions[i] = Quaternion(
                np.array([entry.q_w, entry.q_x, entry.q_y, entry.q_z])
            )
        return quaternions

    def get_time_from_test_id(self, test_id):
        """Load all time fields for a given test ID from the database."""
        entries = self.get_entries_by_test_id(test_id=test_id)
        Nr = len(entries)
        times = [None] * Nr

        for i, entry in enumerate(entries):
            times[i] = entry.time
        return times


def database_worker(engine, model, queue: Queue, stop_event: Event, logger=None):
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        while not stop_event.is_set() or not queue.empty():
            batch = []
            try:
                # Collecte un batch à insérer
                while not queue.empty():
                    batch.append(queue.get(timeout=0.1))
            except Exception:
                pass

            if batch:
                try:
                    session.bulk_save_objects(batch)
                    session.commit()
                except Exception as e:
                    if logger:
                        logger.error(f"[DB WORKER] Registration error: {e}")
                    session.rollback()
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    # Configuration du client IMU
    imu = IMU("IMU_BN055_01", "192.168.0.11", 64344)
    imu.connect()
    # sleep(2)
    # for _ in range(1000):
    #     sleep(0.01)
    #     imu.ping_data()
    # imu.stop_listen_data()
