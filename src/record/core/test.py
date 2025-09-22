"""
Helpers to generate, register, and manage test metadata and patients.

This module provides small classes for:
- Building randomized test plans across ratios/runs/visits (Tests)
- Managing a single test entry and syncing with the DB (Test)
- Maintaining patient entries (Patients)
"""

import numpy as np
import logging
from enum import Enum
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker, Session

from record.core.database import (
    TestDB,
    PatientDB,
    check_for_existing_test_id,
    delete_by_test_id,
)
from datetime import datetime
from record.core.constants import PARIS_TIME_ZONE


class TestStatus(Enum):
    PASSED = "completed"
    FAILED = "failed"
    TODO = "todo"


class Tests:
    """
    Generate a randomized sequence of Test instances and register them.

    Parameters
    ----------
    start_ratio, end_ratio, step : float
        Ratio sweep parameters.
    test_prefix_name : str
        Prefix/name used to build `test_id`.
    sw : float
        Shoulder width used to compute door width (dw = ratio * sw).
    visit : int
        Visit index (0/1 ...).
    engine : sqlalchemy.Engine, optional
        Database engine to bind test metadata.
    additionnal_handler : logging.Handler, optional
        Extra handler to attach to the logger.
    """

    def __init__(
        self,
        start_ratio=0.9,
        end_ratio=2.0,
        step=0.1,
        test_prefix_name="test",
        sw=0.0,
        visit=0,
        engine=None,
        additionnal_handler=None,
    ):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.step = step
        self.test_prefix_name = test_prefix_name
        self.sw = sw
        self.visit = visit
        self.tests = []
        self.engine = engine
        self.additionnal_handler = additionnal_handler
        self.setup_logger()
        self.run()

    def setup_logger(self):
        self.logger = logging.getLogger(f"{Tests.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        formatter = logging.Formatter(
            "[%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s()]: [%(name)s - \033[35m%(message)s\033[0m]"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # if hasattr(self.parent(), "handler"):
        #     self.parent().handler.setFormatter(formatter)
        #     self.logger.addHandler(self.parent().handler)
        if self.additionnal_handler is not None:
            self.additionnal_handler.setFormatter(formatter)
            self.logger.addHandler(self.additionnal_handler)

        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    def run(self):
        """Generate tests and register a TODO entry for each one."""
        self.generate()
        for test in self.tests:
            test.record_setting().record_start()

    @staticmethod
    def linspace(start, stop, step=1.0):
        """
        Like np.linspace but uses step instead of num (inclusive of `stop`).

        Source: https://stackoverflow.com/questions/31820107/is-there-a-numpy-function-that-allows-you-to-specify-start-step-and-number

        Example: start=1, stop=3, step=0.5 -> [1., 1.5, 2., 2.5, 3.]
        """
        return np.linspace(start, stop, int((stop - start) / step + 1))

    def generate(self):
        """
        Build the randomized list of Test instances across ratios and runs.
        Avoids consecutive equal ratios and direct 0.1 jumps.
        """
        raw_ratios = self.linspace(self.start_ratio, self.end_ratio, self.step)
        uratios = [round(a, 2) for a in raw_ratios]
        dratios = {}
        for u in uratios:
            dratios[u] = 0
        ratios = np.repeat(raw_ratios, 3)
        while True:
            np.random.shuffle(ratios)
            g = np.array([round(a, 2) for a in np.diff(ratios)])
            if 0.1 in abs(g):
                continue
            elif 0 in g:  # Check if all elements in g are close to zero
                continue
            else:
                break
        for r in ratios:
            count = dratios[round(r, 2)]
            ratio = round(r, 2)
            test = Test(
                ratio=ratio,
                run=count,
                visit=self.visit,
                sw=self.sw,
                dw=ratio * self.sw,
                test_prefix_name=self.test_prefix_name,
                engine=self.engine,
                additionnal_handler=self.additionnal_handler,
            )
            self.tests.append(test)
            dratios[round(r, 2)] += 1

    @property
    def test_names(self):
        return [t.name for t in self.tests]

    @property
    def test_dict(self):
        d = {t.name: t for t in self.tests}
        return d

    def __repr__(self):
        return f"Tests({self.tests})"

    def __getitem__(self, test_name):
        return self.get_test_by_name(test_name)

    def get_test_by_name(self, name):
        for t in self.tests:
            if t.name == name:
                return t
        raise ValueError(f"Test {name} not found")


class Test:
    """
    Wrapper around a single TESTS entry with convenience DB methods.
    """

    def __init__(
        self,
        ratio,
        sw,
        dw,
        run,
        visit,
        test_prefix_name="test",
        engine=None,
        additionnal_handler=None,
    ):
        self.ratio = ratio
        self.sw = sw
        self.dw = dw
        self.run = run
        self.visit = visit
        self.test_prefix_name = test_prefix_name
        self._status = TestStatus.TODO
        self.engine = engine
        self.additionnal_handler = additionnal_handler
        self.setup_logger()

    def setup_logger(self):
        self.logger = logging.getLogger(f"{Test.__name__}")
        handler = logging.StreamHandler()  # Sortie des logs sur la console
        formatter = logging.Formatter(
            "[%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s()]: [%(name)s - \033[35m%(message)s\033[0m]"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if self.additionnal_handler is not None:
            self.additionnal_handler.setFormatter(formatter)
            self.logger.addHandler(self.additionnal_handler)

        self.logger.setLevel(logging.INFO)  # Niveau par défaut
        self.logger.propagate = False  # Évite la duplication des logs

    @property
    def test_id(self):
        return f"{self.test_prefix_name}_{self.visit}_{self.ratio}_{self.run}"

    @property
    def name(self):
        return self.test_id

    @property
    def status(self):
        """Return the current TestStatus from DB if present, else the in-memory value."""
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        entry = self.session.query(self.model).filter_by(test_id=self.test_id).first()
        self.session.close()
        if entry:
            return TestStatus(entry.status)
        else:
            return self._status

    @status.setter
    def status(self, value):
        if isinstance(value, TestStatus):
            self._status = value
            self.logger.info("Update test metadata in database.")
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            entry = (
                self.session.query(self.model).filter_by(test_id=self.test_id).first()
            )
            if entry:
                entry.status = self._status.value
                self.session.commit()

            else:
                self.logger.error("No test entry found with the given test_id.")
        else:
            raise ValueError("Invalid test status")
        self.session.close()

    def set_start_time(self):
        """
        Set the start time of a test in the database.

        """
        Session = sessionmaker(bind=self.engine)
        session = Session()
        entry = session.query(self.model).filter_by(test_id=self.test_id).first()
        if entry:
            self.logger.info("Add start time to test metadata in database.")
            entry.start_time = datetime.now(PARIS_TIME_ZONE).time()
            session.commit()
        else:
            self.logger.error("No test entry found with the given test_id.")
        session.close()

    def set_end_time(self):
        """
        Set the end time of a test in the database.
        """
        Session = sessionmaker(bind=self.engine)
        session = Session()
        entry = session.query(self.model).filter_by(test_id=self.test_id).first()
        if entry:
            self.logger.info("Add end time to test metadata in database.")
            entry.end_time = datetime.now(PARIS_TIME_ZONE).time()
            session.commit()
        session.close()

    def __repr__(self):
        return f"Test({self.test_id})"

    def record_setting(self, engine=None):
        if not hasattr(self, "engine"):
            self.engine = engine
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.model = TestDB.get_model(tablename="TESTS", class_name="TestDB")
        self.model.metadata.create_all(self.engine)
        return self

    def record_start(self):  # TODO RM old name record_test_metadata
        """
        Add entry to the database with metadata of the test in table TESTS with status 'todo'.
        """
        is_existing = check_for_existing_test_id(self.session, self.model, self.test_id)
        if not is_existing:
            self.logger.info("Add test metada to database.")
            new_entrie = self.model(
                test_id=self.test_id,
                move_id=self.test_prefix_name,
                door_width=self.dw,
                ratio=self.ratio,
                run=self.run,
                visit=self.visit,
                status=TestStatus.TODO.value,
            )
            self.session.add(new_entrie)
        self.session.commit()

    def record_add_test(self):
        """
        Add entry to the database with metadata of the test in table TESTS with status 'todo'.
        """
        is_existing = check_for_existing_test_id(self.session, self.model, self.test_id)
        if not is_existing:
            self.logger.info("Add test metada to database.")
            with Session(self.engine) as session:
                new_entrie = TestDB(
                    test_id=self.test_id,
                    move_id=self.test_prefix_name,
                    door_width=self.dw,
                    ratio=self.ratio,
                    run=self.run,
                    visit=self.visit,
                    status=TestStatus.TODO.value,
                )
                session.add(new_entrie)
                session.commit()

    def record_mark_endtime(self):
        """Update the end_time field of the current test entry to now()."""
        self.logger.info("Recording end time for marker.")
        with Session(self.engine) as session:
            stmt = select(TestDB).where(TestDB.test_id == self.test_id)
            test_entry = session.scalars(stmt).one()
            test_entry.end_time = datetime.now()
            session.commit()

    def check_for_existing_test(self):
        """
        Return True if an entry exists in DB and its status is not TODO.
        """
        self.logger.info("Checking for existing test...")
        is_existing = check_for_existing_test_id(self.session, self.model, self.test_id)
        if is_existing:
            # get value of attribute status
            entry = (
                self.session.query(self.model).filter_by(test_id=self.test_id).first()
            )
            if entry.status != TestStatus.TODO.value:
                return True
        else:
            return False


class Patients:
    """
    Small manager for patient entries stored in the PATIENTS table.
    """

    def __init__(self, engine=None, log_handler=None):
        self.engine = engine
        self.setup_logger()
        if log_handler:
            self.add_handler(log_handler)

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def setup_logger(self):
        """
        Set up the logger for this device.

        This method clears existing handlers, sets a stream handler, defines a formatter,
        adds the handler to the logger, and configures the log level.
        """
        self.logger = logging.getLogger(f"{self.get_class_name()}")
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

    def add_handler(self, handler):
        self.logger.addHandler(handler)

    def add_patient(self, move_id, shoulder_width):
        with Session(self.engine) as session:
            self.logger.info("Check for existing patient entry")
            # Check if patient move_id exists if not create an entry
            if session.query(PatientDB).filter_by(move_id=move_id).first() is None:
                self.logger.info("No existing patient entry found. Creating new one.")
                patient_entry = PatientDB(
                    move_id=move_id, shoulder_width=shoulder_width
                )
                session.add(patient_entry)
                session.commit()

    def get_patients(self):
        with Session(self.engine) as session:
            return session.query(PatientDB).all()

    def get_patient_from_move_id(self, move_id):
        with Session(self.engine) as session:
            return session.query(PatientDB).filter_by(move_id=move_id).first()

    def __getitem__(self, move_id):
        return self.get_patient_from_move_id(move_id)


if __name__ == "__main__":
    from sqlalchemy import create_engine

    engine = create_engine(
        "sqlite:////Users/celmo/Git/record-imucap/src/record/database/database.db"
    )
    patients = Patients(engine=engine)
    patients.add_patient(move_id="M01IV", shoulder_width=42)
