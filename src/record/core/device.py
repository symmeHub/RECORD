"""
Abstract base class for TCP-connected devices with logging and DB helpers.
"""

import socket
import logging
from time import sleep
from abc import ABC, abstractmethod
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from record.core.decorators import threaded
import record.core.database as db
import threading
from colorama import Fore
from colorama import Style


class Device(ABC):
    def __init__(self, name: str, ip_address: str, port: int, label: str = None):
        """
        Initialize a generic device client.

        Parameters
        ----------
        name : str
            Name of the device (used for logging/DB labels).
        ip_address : str
            IP address of the remote server.
        port : int
            Port number of the remote server.
        label : str, optional
            Optional human-readable label.
        """

        self.name = name
        self.ip_address = ip_address
        self.port = port
        self.label = label

        self.TIMEOUT = 1  # 1 second timout
        self.attempt = 0
        self._flag_listen_data = False

        # Logger configuration
        self.logger = logging.getLogger(f"{name}")
        self.setup_logger()

        self.runner_thread = threading.Thread(target=lambda: None, daemon=True)

    def setup_logger(self):
        """
        Set up the logger for this device.

        This method clears existing handlers, sets a stream handler, defines a formatter,
        adds the handler to the logger, and configures the log level.
        """

        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        formatter = logging.Formatter(
            "[%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s()]: [%(name)s - %(message)s]"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    @property
    def is_connected(self):
        return self.runner_thread.is_alive()

    @classmethod
    def init_from_db_engine(cls, name: str, engine):
        """
        Convenience initializer that binds an existing SQLAlchemy engine.
        """
        instance = cls(name=name, ip_address="", port="")
        instance.record_setting(engine)
        return instance

    @staticmethod
    def check_database_url(database_url):
        """
        Normalize a SQLite database URL, adding the `sqlite:///` prefix if missing.
        """
        if not database_url.startswith("sqlite:///"):
            database_url = "sqlite:///" + database_url
        return database_url

    @classmethod
    def init_from_db_url(cls, name: str, database_url: str):
        return cls.init_from_db_engine(
            name=name, engine=create_engine(cls.check_database_url(database_url))
        )

    def connect_db(self, database_url: str):
        self.record_setting(engine=create_engine(self.check_database_url(database_url)))

    def connect(self):
        """
        Establish a connection to the device.

        This method attempts to create a TCP socket and connect to the specified IP address
        and port. If successful, it starts a thread to listen for incoming data.
        """
        try:
            self.logger.info(f"Connection to {self.ip_address}:{self.port}")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.TIMEOUT is not None:
                self.socket.settimeout(self.TIMEOUT)
            self.socket.connect((self.ip_address, self.port))
            self.runner_thread = threading.Thread(target=self._listen_data, daemon=True)
            self.runner_thread.start()
        except Exception as e:
            self.logger.error(f"Erreur de connexion : {e}")
            self.reconnect()

    def reconnect(self, retries=5, delay=5):
        """
        Attempt to reconnect to the device.

        This method retries connecting to the device up to a specified number of attempts
        with a delay between each attempt. If successful, it starts a thread to listen for incoming data.

        :param retries: Maximum number of retry attempts (default is 5).
        :param delay: Delay in seconds between each retry attempt (default is 5 seconds).
        """
        self.logger.info("Attemp to reconnect...")
        self.attempt = 0
        while self.attempt < retries and self._flag_listen_data:
            self.attempt += 1
            try:
                self.logger.info(f"Tentative {self.attempt}/{retries}...")
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.TIMEOUT)
                self.socket.connect((self.ip_address, self.port))
                self.runner_thread = threading.Thread(
                    target=self._listen_data, daemon=True
                )
                self.runner_thread.start()
                return

            except (socket.timeout, ConnectionError) as e:
                self.logger.warning(
                    f"Échec de connexion : {e}. Nouvelle tentative dans {delay} secondes..."
                )
                sleep(delay)
            except Exception as e:
                self.logger.warning(
                    f"Échec de connexion : {e}. Nouvelle tentative dans {delay} secondes..."
                )
                sleep(delay)

    def _listen_data(self):
        """
        Listen for incoming data from the device.

        This method continuously listens for data from the socket and processes it. If an error
        occurs, it attempts to reconnect.
        """
        try:
            self.logger.info(f"{Fore.GREEN}Device connected.{Style.RESET_ALL}")
            self._flag_listen_data = True
            processing_socket_data = self.get_processing_func()
            while self._flag_listen_data:
                processing_socket_data(self)
        except ConnectionError as e:
            self.logger.error(f"Connection interrupted : {e}")
            sleep(0.5)
            self.reconnect()

        except Exception as e:
            self.logger.error(f"Autre erreur : {e}")
            sleep(0.5)
            self.reconnect()

        else:
            pass

    def _listen_data_stop(self):
        """
        Stop listening for data from the device.

        This method sets a flag to stop the listening loop.
        """
        self._flag_listen_data = False

    @abstractmethod
    def get_processing_func(self):
        """
        Return a function that processes incoming socket data.
        This method must be implemented by subclasses to
        define how incoming socket data should be processed.
        """
        pass

    @abstractmethod
    def record_setting(self):
        """
        Set up recording settings for the device.

        This method must be implemented by subclasses to configure recording settings, such as connecting to a database.
        """
        pass

    @abstractmethod
    @threaded
    def record_start(self):
        pass

    @abstractmethod
    def is_data_valid(self) -> bool:
        """
        Check if the data received from the device is valid.
        This method must be implemented by subclasses to define what constitutes valid data for their specific devices.
        """
        return True

    def record_pause_unpause(self):
        """
        Pause or unpause data recording.

        If recording is currently paused, this method resumes it. Otherwise, it pauses the recording.
        """
        if self.flag_record_pause:
            self.flag_record_pause = False
            self.logger.info("Recording Unpaused")
        else:
            self.flag_record_pause = True
            self.logger.info("Recording Paused")

    def record_stop(self):
        """
        Stop data recording.

        This method stops the recording process and commits any remaining records to the database.
        It also logs a message indicating that the recording has stopped and the database has been disconnected.
        """
        self.flag_record_data = False
        self.flag_record_pause = False
        self.logger.info("\033[33mRecording Stopped\033[0m")
        if hasattr(self, "session"):
            self.logger.info("A record session exists")
            self.session.commit()
        # self.engine.dispose()
        self.logger.info("Database Disconnected")
        sleep(0.5)

    def hard_socket_closing(self):
        """
        Close the socket connection.

        This method properly shuts down and closes the socket, and removes the socket attribute.
        """
        if hasattr(self, "socket"):
            self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
            del self.socket

    def close(self):
        """
        Close the device client.

        This method stops recording, stops listening for data, waits briefly to ensure the listener is done,
        and then closes the socket connection.
        """
        try:
            self.record_stop()
            self._listen_data_stop()
            sleep(0.1)  # Give 1 sc to wait until listenning is actual done
            self.hard_socket_closing()
        except Exception as e:
            self.logger.error(f"Failed to close : {e}")
        else:
            return True
        return False

    def stop(self):
        """
        Stop the device client.

        This method logs a message indicating that the device client is stopping and then calls the `close` method.
        """
        self.logger.info("Stopping the device client")
        self.close()

    @property
    def is_recording(self):
        """
        Check if the device client is currently recording data.
        Returns True if recording is in progress, False otherwise.
        """
        return self.flag_record_data

    def create_session(self):
        """
        Create and return a new SQLAlchemy session bound to the engine.
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def get_entries_by_test_id(self, test_id: str, session=None):
        """
        Retrieve entries from the database for a specific test ID.

        Parameters
        ----------
        test_id : str
            Test identifier to filter by.
        session : sqlalchemy.orm.Session, optional
            Existing session; if omitted, a temporary one is created.

        Returns
        -------
        list
            List of ORM instances matching the test ID.
        """
        if hasattr(self, "session"):
            session = self.session
        elif session is None:
            session = self.create_session()

        entries = db.get_entries_by_test_id(session, self.model, test_id)

        if not hasattr(self, "session"):
            session.close()

        return entries

    def get_test_ids(self, session=None):
        """
        Retrieve a list of distinct test IDs from the database.

        Parameters
        ----------
        session : sqlalchemy.orm.Session, optional
            Existing session; if omitted, a temporary one is created.

        Returns
        -------
        list[str]
            Distinct test IDs.
        """
        if hasattr(self, "session"):
            session = self.session
        elif session is None:
            session = self.create_session()

        test_ids = [
            test_id[0] for test_id in session.query(self.model.test_id).distinct().all()
        ]

        if not hasattr(self, "session"):
            session.close()

        return test_ids
