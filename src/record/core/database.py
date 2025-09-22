"""
Database ORM models and lightweight SQLite utilities.

This module defines SQLAlchemy ORM base classes for IMU, trigger, test, and
post-processed tables, along with helper functions to query/delete entries and
an editor class to manipulate SQLite tables (rename, drop, reorder columns, etc.).
"""

import uuid

from datetime import datetime, timezone
import logging
from time import sleep, time
import math
from typing import Any, Optional
from typing import Protocol
from datetime import datetime, timedelta
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    Date,
    Time,
    create_engine,
    String,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.ext.asyncio import AsyncAttrs
from record.core.constants import PARIS_TIME_ZONE
import sqlite3
import logging


_model_cache = {}


class Base(AsyncAttrs, DeclarativeBase):
    @classmethod
    def get_model(cls, tablename: str, class_name: str = None) -> Any:
        """
        Dynamically create (or reuse) a Declarative model for a given table name.

        Parameters
        ----------
        tablename : str
            Target SQL table name.
        class_name : str, optional
            Explicit class name; defaults to the subclass name.

        Returns
        -------
        type
            A SQLAlchemy Declarative model bound to `tablename`.
        """
        if class_name is None:
            class_name = cls.get_classname()

        # if class_name in _model_cache:
        #     return _model_cache[class_name]

        Model = type(class_name, (cls,), {"__tablename__": tablename})
        _model_cache[class_name] = Model
        # print(f"Model {class_name} created for table {tablename}.")

        return Model

    @classmethod
    def get_classname(cls) -> str:
        return cls.__name__

    @staticmethod
    def get_datetime(date: datetime.date, time: datetime.time) -> datetime:
        return datetime.combine(date, time)


class PatientDB(Base):
    __tablename__ = "PATIENTS"
    __table_args__ = {"extend_existing": True}

    move_id: Mapped[String] = Column(
        String,
        unique=True,
        primary_key=True,
    )
    date: Mapped[Date] = Column(Date, default=datetime.now(PARIS_TIME_ZONE).date())
    time: Mapped[Time] = Column(Time, default=datetime.now(PARIS_TIME_ZONE).time())
    shoulder_width: Mapped[Float] = mapped_column(Float)


class TestDB(Base):
    __tablename__ = "TESTS"
    __table_args__ = {"extend_existing": True}

    test_id: Mapped[str] = Column(
        String,
        default=lambda: uuid.uuid4().int >> (128 - 32),
        unique=True,
        primary_key=True,
    )

    date: Mapped[Date] = Column(Date, default=datetime.now(PARIS_TIME_ZONE).date())
    start_time: Mapped[Time] = Column(
        Time, default=datetime.now(PARIS_TIME_ZONE).time()
    )
    end_time: Mapped[Time] = Column(Time, default=datetime.now(PARIS_TIME_ZONE).time())
    move_id: Mapped[String] = mapped_column(String(30))
    door_width: Mapped[Float] = mapped_column(Float)
    ratio: Mapped[Float] = mapped_column(Float)
    run: Mapped[Integer] = mapped_column(Integer)
    visit: Mapped[Integer] = mapped_column(Integer)
    status: Mapped[String] = mapped_column(
        String(30), default="todo"
    )  # new field to track the status of the test can be "todo", "completed" or "failed"

    def __repr__(self):
        repr_str = f"TestDB(id={self.test_id})"
        return repr_str


class IMUPacketDB(Base):
    __abstract__ = True
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    test_id: Mapped[str] = Column(String(30))
    date: Mapped[Date] = Column(Date, default=datetime.now(PARIS_TIME_ZONE).date())
    time: Mapped[Time] = Column(Time, default=datetime.now(PARIS_TIME_ZONE).time())
    imu_elapsed_time: Mapped[Optional[float]] = mapped_column(
        Float, doc="Elapsed time in microseconds"
    )
    imu_elapsed_time_unit: Mapped[str] = mapped_column(String(10))

    q_w: Mapped[Float] = mapped_column(Float)
    q_x: Mapped[Float] = mapped_column(Float)
    q_y: Mapped[Float] = mapped_column(Float)
    q_z: Mapped[Float] = mapped_column(Float)

    acc_x: Mapped[Optional[float]] = mapped_column(Float, doc="Acceleration in x-axis")
    acc_y: Mapped[Optional[float]] = mapped_column(Float, doc="Acceleration in y-axis")
    acc_z: Mapped[Optional[float]] = mapped_column(Float, doc="Acceleration in z-axis")

    def __repr__(self):
        repr_str = f"<DataPacketDB(id={self.id},\ttimestamp={self.timestamp}\r"
        repr_str += f"\timu_elapsed_time={self.imu_elapsed_time} [{self.imu_elapsed_time_unit}]\r"
        repr_str += (
            f"\tq_x={self.q_x:.2f}, \tq_y={self.q_y:.2f}, \tq_z={self.q_z:.2f}\r"
        )
        return repr_str


class PostProcessedIMUDB(Base):
    __abstract__ = True
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True)
    # id_raw: Mapped[int] = mapped_column(Integer)
    test_id: Mapped[str] = Column(String(30))
    yaw: Mapped[float] = Column(Float)
    pitch: Mapped[float] = Column(Float)
    roll: Mapped[float] = Column(Float)
    norm_acc: Mapped[float] = Column(Float)
    delta_time: Mapped[float] = Column(Float)
    zone: Mapped[str] = Column(String(30))
    inflexion_point: Mapped[bool] = Column(Boolean)


class PostProcessedTestDB(Base):
    __tablename__ = "POST_PROCESSED_TESTS"
    __table_args__ = {"extend_existing": True}

    test_id: Mapped[str] = Column(
        String,
        default=lambda: uuid.uuid4().int >> (128 - 32),
        unique=True,
        primary_key=True,
    )

    sway_diff: Mapped[Float] = mapped_column(Float, nullable=True)
    mean_sway_zone_1: Mapped[Float] = mapped_column(Float, nullable=True)
    max_sway_zone_2: Mapped[Float] = mapped_column(Float, nullable=True)
    mean_norm_acc_zone_1: Mapped[Float] = mapped_column(Float, nullable=True)
    mean_norm_acc_zone_2: Mapped[Float] = mapped_column(Float, nullable=True)
    order: Mapped[Integer] = mapped_column(Integer, nullable=True)

    def __repr__(self):
        repr_str = f"PostProcessedTestDB(id={self.test_id})"
        return repr_str


class TriggerPacketDB(Base):
    __abstract__ = True
    __table_args__ = {"extend_existing": True}

    test_id: Mapped[str] = Column(
        String,
        default=lambda: uuid.uuid4().int >> (128 - 32),
        unique=True,
        primary_key=True,
    )
    date: Mapped[Date] = Column(Date, default=datetime.now(PARIS_TIME_ZONE).date())
    time: Mapped[Time] = Column(Time, default=datetime.now(PARIS_TIME_ZONE).time())
    state: Mapped[Boolean] = mapped_column(Boolean)


def check_for_existing_test_id(session, model, test_id):
    """
    Return the first entry for `test_id` if it exists, else None.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
    model : Declarative model
    test_id : str
    """
    is_entry_exist = session.query(model).filter_by(test_id=test_id).first()
    return is_entry_exist


def delete_by_test_id(session, model, test_id):
    """
    Delete all rows matching a given `test_id`.
    """
    try:
        entries_to_delete = session.query(model).filter_by(test_id=test_id).all()
        if len(entries_to_delete) > 0:
            for entry in entries_to_delete:
                session.delete(entry)

            session.commit()
            print(
                f"Enregistrement avec test_id '{test_id}' supprimé de la table {model.__tablename__}."
            )

        else:
            print(
                f"Aucun enregistrement trouvé avec test_id '{test_id}' dans la table {model.__tablename__}."
            )
    except Exception as e:
        session.rollback()
        print(f"Une erreur s'est produite lors de la suppression : {e}")


def get_entries_by_test_id(session, model, test_id):
    """
    Fetch all rows matching a given `test_id`.
    Returns a list; returns [] on error.
    """
    try:
        entries = session.query(model).filter_by(test_id=test_id).all()
        return entries
    except Exception as e:
        print(
            f"Une erreur s'est produite lors de la récupération des enregistrements : {e}"
        )
        return []
    finally:
        session.close()


def get_attr_entries_from_model(attr: str, session, model):
    """
    Collect a single attribute from all rows for a given model.
    """
    attrs = []
    try:
        # attrs = [entry.test_id for entry in session.query(model).all()]
        for entry in session.query(model).all():
            attrs.append(getattr(entry, attr))
    except Exception as e:
        print(
            f"Une erreur s'est produite lors de la récupération des IDs de tests : {e}"
        )
        return []
    finally:
        session.close()
        return attrs


class SQLiteTableEditor:
    """
    Helper to perform structural changes on a SQLite table by rebuilding it.

    This class uses the typical SQLite workflow of creating a temporary table,
    copying/translating rows, and replacing the original table, enabling
    operations like renaming, dropping, duplicating and reordering columns.
    """

    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _get_columns_info(self, cursor):
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        return cursor.fetchall()

    def get_column_order(self):
        """
        Return the current list of column names in order.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            return [col[1] for col in self._get_columns_info(cursor)]

    def _rebuild_table(self, new_columns, transform_func=lambda row: row):
        """
        Create a new table with `new_columns`, migrate data using `transform_func`,
        then replace the original table.
        """
        conn = self._connect()
        cursor = conn.cursor()

        old_columns_info = self._get_columns_info(cursor)
        old_column_names = [col[1] for col in old_columns_info]

        temp_table = f"{self.table_name}_temp"

        # Créer la table temporaire
        create_stmt = f"CREATE TABLE {temp_table} ({', '.join(new_columns)})"
        cursor.execute(create_stmt)

        # Récupérer les données existantes
        cursor.execute(f"SELECT * FROM {self.table_name}")
        data = cursor.fetchall()

        # Insérer les données transformées dans la nouvelle table
        for row in data:
            new_row = transform_func(dict(zip(old_column_names, row)))
            cursor.execute(
                f"INSERT INTO {temp_table} ({', '.join([col.split()[0] for col in new_columns])}) VALUES ({', '.join(['?'] * len(new_columns))})",
                new_row,
            )

        # Supprimer l’ancienne table et renommer la nouvelle
        cursor.execute(f"DROP TABLE {self.table_name}")
        cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {self.table_name}")

        conn.commit()
        conn.close()

    def rename_column(self, old_name, new_name):
        """
        Rename a column, preserving order and types.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            columns_info = self._get_columns_info(cursor)

            old_column_names = [col[1] for col in columns_info]
            column_types = {col[1]: col[2] for col in columns_info}

            if old_name not in old_column_names:
                raise ValueError(f"La colonne '{old_name}' n'existe pas dans la table.")

            # Nouveau nom des colonnes (ordre conservé)
            new_column_names = [
                new_name if col == old_name else col for col in old_column_names
            ]
            new_definitions = [
                f"{new_col} {column_types[old_col]}"
                for old_col, new_col in zip(old_column_names, new_column_names)
            ]

        def transform(row):
            return [
                row[old_name] if new_col == new_name else row[new_col]
                for old_col, new_col in zip(old_column_names, new_column_names)
            ]

        self._rebuild_table(new_definitions, transform)

    def drop_column(self, column_name):
        """
        Drop a column by name.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            columns_info = self._get_columns_info(cursor)
            new_columns = [
                f"{col[1]} {col[2]}" for col in columns_info if col[1] != column_name
            ]
            column_order = [col[1] for col in columns_info if col[1] != column_name]

        self._rebuild_table(new_columns, lambda row: [row[col] for col in column_order])

    def add_column(self, column_name, column_type, default=None):
        """
        Add a new column with optional default value.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            default_clause = f" DEFAULT {repr(default)}" if default is not None else ""
            cursor.execute(
                f"ALTER TABLE {self.table_name} ADD COLUMN {column_name} {column_type}{default_clause}"
            )
            conn.commit()

    def duplicate_column(self, source_column, new_column):
        """
        Create a new column as a duplicate of an existing one.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            columns_info = self._get_columns_info(cursor)

            # Vérifie si la colonne source existe
            types = {col[1]: col[2] for col in columns_info}
            if source_column not in types:
                raise ValueError(f"La colonne '{source_column}' n'existe pas.")

            new_columns = [f"{col[1]} {col[2]}" for col in columns_info] + [
                f"{new_column} {types[source_column]}"
            ]
            column_order = [col[1] for col in columns_info]

        self._rebuild_table(
            new_columns,
            lambda row: [row[col] for col in column_order] + [row[source_column]],
        )

    def reorder_columns(self, new_order):
        """
        Reorder columns to match `new_order` exactly (same set of columns required).
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            columns_info = self._get_columns_info(cursor)

            existing_columns = {col[1]: col[2] for col in columns_info}
            if set(new_order) != set(existing_columns):
                raise ValueError(
                    "La liste fournie doit contenir exactement les mêmes colonnes que la table actuelle."
                )

            new_definitions = [f"{col} {existing_columns[col]}" for col in new_order]

        self._rebuild_table(
            new_definitions, lambda row: [row[col] for col in new_order]
        )

    def add_seconds_to_time_column(self, column_name, seconds_to_add):
        """
        Add a number of seconds to a TEXT time column formatted as HH:MM:SS.ffffff.
        """
        with self._connect() as conn:
            cursor = conn.cursor()

            # Vérifier que la colonne existe
            column_names = [col[1] for col in self._get_columns_info(cursor)]
            if column_name not in column_names:
                raise ValueError(f"Colonne '{column_name}' introuvable.")

            # Récupérer les identifiants + temps actuels
            cursor.execute(f"SELECT rowid, {column_name} FROM {self.table_name}")
            rows = cursor.fetchall()

            for rowid, time_str in rows:
                try:
                    original_time = datetime.strptime(time_str, "%H:%M:%S.%f")
                    new_time = original_time + timedelta(seconds=seconds_to_add)
                    new_time_str = new_time.strftime("%H:%M:%S.%f")

                    cursor.execute(
                        f"UPDATE {self.table_name} SET {column_name} = ? WHERE rowid = ?",
                        (new_time_str, rowid),
                    )
                except Exception as e:
                    print(f"Erreur avec rowid {rowid} : {e}")

            conn.commit()


if __name__ == "__main__":
    engine = create_engine("sqlite:////Users/celmo/Git/record/assets/database_test.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    model = TriggerPacketDB.get_model(tablename="TRIGGER_SENSOR_00")
    model.metadata.create_all(engine)

    # session.query(model).filter_by(test_id='test_0.9_0').first()
    # my_test_db = TestDB.get_model(tablename="TESTS", class_name="TestDB")
    # my_test_db.metadata.create_all(engine)

    # new_entrie = my_test_db(
    #     move_id="toto", shoulder_width=60.0, door_width=100.0, ratio=1.1, run=2
    # )

    # session.add(new_entrie)
    # session.commit()
