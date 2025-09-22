"""
Post-processing pipeline for IMU and trigger data stored in the database.

Includes utilities to compute Euler angles, smoothing, segmentation into
zones, peak/valley detection, and sway metrics, as well as orchestration
classes to run post-processing per test or per patient.
"""

# from record import core
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from sqlalchemy import create_engine, MetaData, Table, delete, inspect
from sqlalchemy.orm import sessionmaker, Session
from record.core.geometry import Quaternion
from scipy.signal import savgol_filter
from scipy import interpolate
from findpeaks import findpeaks, stats
import datetime
from sklearn.cluster import DBSCAN
from record.core.database import (
    TestDB,
    TriggerPacketDB,
    PostProcessedIMUDB,
    PostProcessedTestDB,
)
from record.core.test import Patients
import matplotlib.pyplot as plt


def compute_dt(time_0: datetime.datetime, time_1: datetime.datetime):
    """
    Computes the difference between two times.
    return type : datetime.timedelta
    """
    reference_date = datetime.date.today()
    t0 = datetime.datetime.combine(reference_date, time_0)
    t1 = datetime.datetime.combine(reference_date, time_1)
    return t1 - t0


def get_test_ids(database_url: str):
    _, _, session = AbstractPostProcessing.set_up_database(database_url)

    return [test_id for test_id, in session.query(TestDB.test_id).all()]


class AbstractPostProcessing(ABC):
    def __init__(self, database_url):
        if not database_url.startswith("sqlite:///"):
            database_url = "sqlite:///" + database_url
        self.database_url = database_url
        self.engine, self.metadata, self.session = self.set_up_database(
            database_url=database_url
        )
        self.setup_logger()

    @staticmethod
    def set_up_database(database_url: str):
        engine = create_engine(database_url)
        metadata = MetaData()
        metadata.reflect(bind=engine)
        session = sessionmaker(bind=engine)()
        return engine, metadata, session

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

    @classmethod
    def init_from_engine(cls, engine, *arg, **kwargs):
        return cls(database_url=engine.url.database, *arg, **kwargs)

    @property
    def tables(self) -> [str]:
        return self.get_tables_names()

    def get_tables_names(self) -> [str]:
        return [tablename for tablename in self.metadata.tables.keys()]

    def get_table(self, tablename: str):
        return self.metadata.tables[tablename]

    def get_dataframe(self, tablename: str, filter_dict: dict = {}):
        return pd.read_sql(
            self.session.query(self.get_table(tablename))
            .filter_by(**filter_dict)
            .statement,
            self.session.bind,
        )

    def delete_entries(self, filter_dict: dict = {}):
        """
        Deletes entries in every table of the database that match filter_dict.
        """
        try:
            inspector = inspect(self.engine)

            for tablename in self.metadata.tables.keys():
                columns = inspector.get_columns(tablename)
                column_names = [column["name"] for column in columns]

                # Keep only filters that match existing columns
                valid_filter = {
                    k: v for k, v in filter_dict.items() if k in column_names
                }

                if not valid_filter:
                    continue  # Skip if no valid filters for this table

                table = self.get_table(tablename)
                delete_query = self.session.query(table).filter_by(**valid_filter)

                if delete_query.count() > 0:
                    delete_query.delete(synchronize_session=False)
                    self.logger.info(
                        f"Deleted entries from {tablename} where {valid_filter}"
                    )

            self.session.commit()
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error deleting entries: {e}")

    @abstractmethod
    def run_postprocessing(self):
        """
        This method should be implemented by subclasses to perform postprocessing tasks.
        It should take no arguments and return None.
        It should perform any necessary data analysis or transformation on the data stored in the database.
        It should also save the results of any analysis or transformation to a new table in the database.
        """

        pass


class PostProcessing(AbstractPostProcessing):
    def run_postprocessing(self):
        pass


class IMUObject(PostProcessing):
    def __init__(self, database_url: str, tablename: str, filter_dict: dict = {}):
        super().__init__(database_url=database_url)
        self.tablename = tablename
        self.filter = filter_dict
        self.data_post = self.init_data_post()  # a copy of the that can be write
        # self.compute_euler_angles()
        # self.data_post_quaternions = self.data_post[[c for c in self.data_post.columns if c.startswith("q_")]]
        self.setup_logger()

        self.test_start_time = None
        self.test_end_time = None
        self.test_first_check_time = None
        self.test_second_check_time = None
        self.is_usable = self.data_integrity()

    def run_postprocessing(self, compensation_index=0, standalone=False):
        df = self.load_from_database()
        if len(df) == 0:
            self.add_postprocess_columns()
            self.compute_delta_times()
            self.compute_norm_acc()
            self.apply_compensation(compensation_index)
            self.compute_euler_angles(
                smoothed=True, smooth_window=20, smooth_polyorder=4
            )
            if not standalone:
                self.data_segmentation()
                self.detection_segmented_peaks_valleys()
                self.remove_outliers_inflexion_points_from_zone_2()
            else:
                self.detection_segmented_peaks_valleys()
            self.dump_to_database()
        else:
            self.data_post = df.merge(self.data_post[["id", "time"]], on="id")

    def data_integrity(self):
        # Check if the data is complete and consistent
        # check data type and do conversion if needed

        # Check if any data related to self.filter exists
        Nd = len(self.data)
        try:
            if Nd < 1:
                raise ValueError("No data found in the dataset")
            else:
                if type(self.data_post.time.values[0]) == str:
                    # Conversion to datetime.time type
                    self.logger.info("Converting time data to datetime")
                    # Assuming the time is in 'HH:MM:SS.micros' format
                    time_series = self.data_post.time.apply(
                        lambda x: datetime.datetime.strptime(x, "%H:%M:%S.%f").time()
                    )
                    self.data_post.time = time_series
            return True
        except Exception as e:
            # self.logger.warning(f"Warning: {e}")
            return False

    def update_attributes_from_test(self, attributes: dict):
        self.__dict__.update(attributes)

    @staticmethod
    def get_quaternions_dataframe_from_df(df):
        return df[[c for c in df.columns if c.startswith("q_")]]

    def get_quaternions_dataframe(self):
        return self.get_quaternions_dataframe_from_df(self.data)

    def get_acc_dataframe(self):
        return self.data[[c for c in self.data.columns if c.startswith("acc_")]].apply(
            lambda x: round(x, 2)
        )

    def get_angle_df(
        self,
        angle_type: str,
        zone="all",
        smoothed=False,
        smooth_window=10,
        smooth_polyorder=4,
    ):
        """
        zone : str can be "all", "zone_1", "zone_2"
        """

        if zone == "all":
            angle_series = self._compute_angle_from_quat_df(
                self.data_post_quaternions, angle_type
            )
            self.data_post[angle_type] = np.array([np.nan] * len(angle_series))
            for i in range(len(angle_series)):
                self.data_post[angle_type].loc[i] = angle_series[i]
        else:
            s, e = self.index_zone[zone]
            angle_series = self._compute_angle_from_quat_df(
                self.data_post_quaternions.loc[s:e], angle_type
            )

        if smoothed:
            self.logger.info(
                "{} -- {} -- smoothed, smooth_window={}, smooth_polyorder={}".format(
                    self.tablename, angle_type, smooth_window, smooth_polyorder
                )
            )
            return pd.Series(
                savgol_filter(angle_series, smooth_window, smooth_polyorder),
                name=angle_type,
            )
        else:
            return angle_series

    @staticmethod
    def _compute_angle_from_quat_df(quaternions_dataframe, angle_type: str):
        ANGLE_METHODS_MAP = {
            "yaw": lambda q: np.degrees(q.yaw),
            "pitch": lambda q: np.degrees(q.pitch),
            "roll": lambda q: np.degrees(q.roll),
        }
        if angle_type not in ANGLE_METHODS_MAP:
            raise ValueError(f"Angle {angle_type} not supported")
        angle_func = ANGLE_METHODS_MAP[angle_type]
        angle_series = (
            quaternions_dataframe.apply(lambda x: Quaternion(x.values), axis=1)
            .apply(angle_func)
            .rename(angle_type)
        )
        return angle_series

    def init_data_post(self):
        df = self.data.copy()
        # add empty columns yaw, pitch, roll, delta_time, zone, inflexion_point
        return df

    def add_postprocess_columns(self):
        # add empty columns yaw, pitch, roll, delta
        self.data_post["yaw"] = np.nan
        self.data_post["pitch"] = np.nan
        self.data_post["roll"] = np.nan
        self.data_post["norm_acc"] = np.nan
        self.data_post["delta_time"] = np.nan
        self.data_post["zone"] = ""
        self.data_post["inflexion_point"] = False

    def init_data_post_quaternions(self):
        return self.get_quaternions_dataframe_from_df(self.data_post)

    def compute_euler_angles(
        self,
        zone="all",
        smoothed=True,
        smooth_window=10,
        smooth_polyorder=4,
    ):
        euler_angles = ["yaw", "pitch", "roll"]
        for angle_type in euler_angles:
            self.get_angle_df(
                angle_type,
                zone=zone,
                smoothed=smoothed,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
            )
        return self

    def apply_compensation(self, qoffset_index=-1):
        if qoffset_index == -1:
            return self
        else:
            self.logger.info("Applying compensation for IMU offset. Please wait...")
            qoffset = self.quaternions_df.apply(lambda x: Quaternion(x.values), axis=1)[
                qoffset_index
            ]
            quaternion_series = self.data_post_quaternions.apply(
                lambda x: Quaternion(x.values), axis=1
            ).apply(lambda x: x.q_inv * qoffset)
            self.update_data_post_from_quaternions_series(quaternion_series)
            return self

    def update_data_post_from_quaternions_series(self, quaternion_series):
        for i in range(len(quaternion_series)):
            # lets update self.data_post too
            self.data_post.at[i, "q_w"] = quaternion_series[i]._q[0]
            self.data_post.at[i, "q_x"] = quaternion_series[i]._q[1]
            self.data_post.at[i, "q_y"] = quaternion_series[i]._q[2]
            self.data_post.at[i, "q_z"] = quaternion_series[i]._q[3]

    @property
    def data(self):
        df = self.get_dataframe(self.tablename, self.filter)
        return df[[c for c in df.columns]]

    @property
    def test_id(self):
        return self.filter["test_id"]

    @property
    def data_post_quaternions(self):
        # a view of self.data_post that only includes columns starting with "q_"
        return self.data_post[[c for c in self.data_post.columns if c.startswith("q_")]]

    @property
    def quaternions_df(self):
        return self.get_quaternions_dataframe()

    @property
    def acc_df(self):
        return self.get_acc_dataframe()

    @property
    def acc(self):
        return self.acc_df.values.astype(np.float64)

    @property
    def norm_acc(self):
        return savgol_filter(np.linalg.norm(self.acc, axis=1), 10, 3)

    @property
    def quaternions(self):
        return self.quaternions_df.values.astype(np.float64)

    @property
    def times_df(self):
        return self.data_post.time

    @property
    def times(self):
        return self.times_df.values

    @property
    def start_time(self):
        return self.times_df.min()

    @property
    def end_time(self):
        return self.times_df.max()

    def get_delta_times(self):
        """
        return type : list of time.timedelta

        """
        start_time = self.start_time
        if self.test_start_time is not None:
            start_time = self.test_start_time
        return [PostProcessingTest.compute_dt(start_time, t) for t in self.times]

    def was_imu_disconnected(self, treshold=10) -> bool:
        """
        Perform cluster analysis on elapsed time to detect if the IMU was disconnected.
        treshold: threshold of the groupe size of zeros detect in cluster
        """
        Y = np.abs(np.diff(self.data.imu_elapsed_time.values.astype(np.int64)))
        data = pd.DataFrame({"X": np.arange(len(Y)), "Y": Y})

        # Appliquer DBSCAN pour détecter des groupes avec des valeurs nulles
        dbscan = DBSCAN(eps=5, min_samples=2)  # Ajustez `eps` selon vos données
        data["Cluster"] = dbscan.fit_predict(data)

        # Filtrer les groupes contenant des valeurs nulles
        zero_clusters = data[data["Y"] == 0]["Cluster"].value_counts()

        # Vérifier si l'un de ces clusters a une taille supérieure au seuil
        if any(zero_clusters > treshold):
            return True
        else:
            return False

    @staticmethod
    def clean_extremas(df):
        # Filtrer les lignes où valley ou peak est True
        extrema = df[(df["valley"] == True) | (df["peak"] == True)].copy()

        # Ajouter une colonne type : 'valley' ou 'peak'
        extrema["type"] = extrema.apply(
            lambda row: "valley" if row["valley"] else "peak", axis=1
        )

        # Liste pour stocker les bons indices
        result = []

        # Initialiser avec le premier extremum
        prev_type = None

        i = 0
        while i < len(extrema):
            group = [extrema.iloc[i]]
            current_type = group[0]["type"]
            i += 1

            # Regrouper les extrêmes consécutifs de même type
            while i < len(extrema) and extrema.iloc[i]["type"] == current_type:
                group.append(extrema.iloc[i])
                i += 1

            # Choisir le meilleur dans le groupe
            group_df = pd.DataFrame(group)

            if current_type == "valley":
                selected = group_df.loc[group_df["y"].idxmin()]
            else:  # peak
                selected = group_df.loc[group_df["y"].idxmax()]

            result.append(selected)

        # Créer le DataFrame final alterné
        clean_extrema = pd.DataFrame(result)

        # Facultatif : réindexer proprement
        clean_extrema = clean_extrema.sort_values(by="x").reset_index(drop=True)
        return clean_extrema

    @staticmethod
    def get_peaks_valleys(arr: np.ndarray, lookahead=5):
        fp = findpeaks(method="peakdetect", lookahead=lookahead, interpolate=4)
        results = fp.fit(arr)
        return results["df"]

    @staticmethod
    def get_inflexion_points(df):
        inflection_loc, inflextion_value = df[
            (df["valley"] == True) | (df["peak"] == True)
        ][["x", "y"]].values.T

        return inflection_loc, inflextion_value

    def compute_delta_times(self):
        # update delta time to current test
        self.data_post["delta_time"] = [
            dt.total_seconds() for dt in self.get_delta_times()
        ]

    def data_segmentation(self):
        """
        Segmentation of the data
        1rst groupe of data are beetween self.test_start_time and self.test_first_check_time -> data zone 1
        2nd groupe of data are beetween self.test_second_check_time and self.test_end_time -> data zone 2
        """
        self.logger.info(f"{self.tablename} - Data segmentation started")

        # Extration of sub_df for data zone 1
        self.index_zone = {}
        data_zone_1 = self.data_post[
            (self.data_post["time"] >= self.test_start_time)
            & (self.data_post["time"] <= self.test_first_check_time)
        ]
        data_zone_2 = self.data_post[
            (self.data_post["time"] >= self.test_second_check_time)
            & (self.data_post["time"] <= self.test_end_time)
        ]
        self.index_zone.update(
            {"zone_1": (int(data_zone_1.iloc[0].name), int(data_zone_1.iloc[-1].name))}
        )
        self.index_zone.update(
            {"zone_2": (int(data_zone_2.iloc[0].name), int(data_zone_2.iloc[-1].name))}
        )

        # Lets add a column zone to data to identify which zone each row belongs to
        self.data_post["zone"] = len(self.data) * ["Unknown"]
        self.data_post.loc[
            self.index_zone["zone_1"][0] : self.index_zone["zone_1"][1] + 1, "zone"
        ] = "zone_1"
        self.data_post.loc[
            self.index_zone["zone_2"][0] : self.index_zone["zone_2"][1] + 1, "zone"
        ] = "zone_2"

    def detection_segmented_peaks_valleys(self, lc=0.10):
        """
        Compute the peaks and valleys for each zone of data.
        lc : lookahead coefficient
        """
        self.logger.info(f"{self.tablename} - Computing zone peaks and valleys started")
        self.data_post["inflexion_point"] = self.data_post.shape[0] * [False]
        # MIN_INF_LOC_SIZE = {
        #     "zone_1":3,
        #     "zone_2":3,
        # }
        for zone in self.data_post.zone.unique():
            zone_yaw = self.data_post[self.data_post.zone == zone]["yaw"].values.astype(
                np.float64
            )

            zone_delta_time = self.data_post[self.data_post.zone == zone][
                "delta_time"
            ].values.astype(np.float64)

            y_meas = np.append(zone_yaw, 0.0)
            x_meas = np.append(zone_delta_time, zone_delta_time[-1] + 0.25)
            Y = interpolate.interp1d(x_meas, y_meas, kind="nearest")
            N_interp = 10000
            x_interp = np.linspace(x_meas.min(), x_meas.max(), num=N_interp)
            y_interp = Y(x_interp)
            zone_inf_loc = np.array([])
            lookahead = int(lc * N_interp)
            while zone_inf_loc.size < 3:
                df = IMUObject.get_peaks_valleys(y_interp, lookahead=lookahead)
                df = IMUObject.clean_extremas(df)
                zone_inf_loc, zone_inf_val = IMUObject.get_inflexion_points(df)
                lookahead /= 2
                lookahead = int(lookahead)
                if lookahead < 10:
                    break

            zone_inf_loc = zone_inf_loc[1:-1].astype(int)

            # Seek for the nest delta_time value  and its corresponding row index
            for loc in zone_inf_loc:
                row = self.data_post.loc[
                    (self.data_post["delta_time"] - x_interp[loc]).abs().argsort()[:1]
                ]

                self.data_post.loc[
                    row.index,
                    "inflexion_point",
                ] = True

    def remove_outliers_inflexion_points_from_zone_2(self):
        zone = "zone_2"

        # Filter rows corresponding to zone_2 and inflexion points
        mask = (self.data_post["zone"] == zone) & (
            self.data_post["inflexion_point"] == True
        )
        yaws = self.data_post.loc[mask, "yaw"].reset_index(drop=True)

        if len(yaws) < 2:
            return  # Not enough points to compute diff

        # Identify the two indices to keep based on max difference
        diffs = np.abs(np.diff(yaws))
        max_diff_idx = np.argmax(diffs)

        ilocs_to_keep = {max_diff_idx, max_diff_idx + 1}
        all_ilocs = set(range(len(yaws)))
        ilocs_to_remove = sorted(all_ilocs - ilocs_to_keep)

        # Get actual indices in the original DataFrame
        indices_to_remove = self.data_post.loc[mask].iloc[ilocs_to_remove].index

        # Set inflexion_point to False for those rows
        self.data_post.loc[indices_to_remove, "inflexion_point"] = False

    @staticmethod
    def compute_angle_from_quat_df(quaternions_dataframe, angle_type: str, angle_func):
        return (
            quaternions_dataframe.apply(lambda x: Quaternion(x.values), axis=1)
            .apply(angle_func)
            .rename(angle_type)
        )

    def compute_zones_sway(self):
        data = {}
        data.update({"zone_1": self.mean_sway_zone(zone="zone_1")})
        data.update({"zone_2": self.max_sway_zone(zone="zone_2")})
        return data

    def mean_sway_zone(self, zone: str):
        """
        zone : str can be "zone_1", "zone_2"
        """
        yaws = self.data_post[
            (self.data_post.zone == zone) & (self.data_post.inflexion_point == True)
        ].yaw
        return np.mean(np.abs(np.diff(yaws)))

    def max_sway_zone(self, zone: str):
        """
        zone : str can be "zone_1", "zone_2"
        """
        try:
            yaws = self.data_post[
                (self.data_post.zone == zone) & (self.data_post.inflexion_point == True)
            ].yaw
            return np.max(np.abs(np.diff(yaws)))
        except ValueError:
            print("No data found for the specified zone.")
            return 0

    def stat_norm_acc_zone(self, func_stat, zone: str):
        norms_acc = self.data_post[
            (self.data_post.zone == zone) & (self.data_post.inflexion_point == True)
        ].norm_acc
        return func_stat(norms_acc)

    def mean_norm_acc_zone(self, zone: str):
        return self.stat_norm_acc_zone(np.mean, zone)

    def max_norm_acc_zone(self, zone: str):
        return self.stat_norm_acc_zone(np.max, zone)

    def compute_norm_acc(self):
        acc = self.data_post[
            [c for c in self.data_post.columns if c.startswith("acc_")]
        ]
        norms_acc = acc.apply((lambda x: np.linalg.norm(x)), axis=1)

        self.data_post["norm_acc"] = norms_acc.values.astype(np.float64)

    def compute_sway_diff(self):
        """
        Compute the difference between the sway of two zones.
        :return: The difference between the sway of two zones.
        """

        data = self.compute_zones_sway()
        return abs(data["zone_2"] - data["zone_1"])

    @property
    def sway_diff(self):
        return self.compute_sway_diff()

    def plot_zones_yaw_inflexions(self, hold=False):
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 8))

        for zone in self.data_post.zone.unique():
            data_zone = self.data_post[self.data_post.zone == zone]
            yaw_y = data_zone.yaw.values
            yaw_x = data_zone.delta_time.values
            yaw_inflextion_y = data_zone[data_zone.inflexion_point == True].yaw.values
            yaw_inflextion_x = data_zone[
                data_zone.inflexion_point == True
            ].delta_time.values
            ax1.plot(yaw_x, yaw_y, ".", label=f"yaw_{zone}")
            ax1.plot(
                yaw_inflextion_x, yaw_inflextion_y, "o", label=f"inflexion point {zone}"
            )

        ax1.grid("on")
        ax1.legend(loc="upper left")
        ax1.set_ylabel("angle [°]")
        ax1.set_ylim(-180, 180)
        ax1.set_xlabel("time [s]")
        ax1.set_title(f"TEST_ID: {self.test_id}")
        if not hold:
            plt.show()

    def load_from_database(self):
        tablename = f"POST_PROCESSED_{self.tablename}"
        model = PostProcessedIMUDB.get_model(
            tablename=tablename, class_name="PostProcessedIMUDB"
        )

        # Ouvre une session
        session = Session(bind=self.engine)

        try:
            # Requête tous les enregistrements et transforme en DataFrame
            query = session.query(model).filter_by(test_id=self.test_id)
            df = pd.read_sql(query.statement, session.bind)
            return df

        except Exception as e:
            print(f"Erreur lors de la lecture depuis la base de données : {e}")
            return pd.DataFrame()  # DataFrame vide en cas d'erreur

        finally:
            session.close()

    def dump_to_database(self):
        tablename = f"POST_PROCESSED_" + self.tablename
        model = PostProcessedIMUDB.get_model(
            tablename=tablename, class_name="PostProcessedIMUDB"
        )
        model.metadata.create_all(bind=self.engine)
        session = Session(bind=self.engine)
        try:
            # Crée une liste d'objets à partir des lignes du DataFrame
            records = [
                model(
                    id=row["id"],
                    test_id=row["test_id"],
                    yaw=row["yaw"],
                    pitch=row["pitch"],
                    roll=row["roll"],
                    delta_time=row["delta_time"],
                    norm_acc=row["norm_acc"],
                    zone=row["zone"],
                    inflexion_point=row["inflexion_point"],
                )
                for _, row in self.data_post.iterrows()
            ]

            # Ajoute tous les objets d'un coup
            session.add_all(records)
            session.commit()

        except Exception as e:
            session.rollback()
            print(f"Erreur lors de l'insertion dans la base de données : {e}")

        finally:
            session.close()


class TestObject(PostProcessing):
    def __init__(self, database_url: str, tablename: str, filter_dict: dict = {}):
        super().__init__(database_url=database_url)
        self.tablename = tablename
        self.filter = filter_dict
        self.model = TestDB.get_model(tablename=self.tablename)
        self._entrie = self.session.query(self.model).filter_by(**self.filter).one()

    @property
    def data(self):
        df = self.get_dataframe(self.tablename, self.filter)
        return df[[c for c in df.columns]]

    @property
    def test_id(self):
        return self._entrie.test_id

    @property
    def move_id(self):
        return self._entrie.move_id

    @property
    def shoulder_width(self):
        return self._entrie.shoulder_width

    @property
    def door_width(self):
        return self._entrie.door_width

    @property
    def ratio(self):
        return self._entrie.ratio

    @property
    def run(self):
        return self._entrie.run

    @property
    def visit(self):
        return self._entrie.visit

    @property
    def timestamp(self):
        return self._entrie.timestamp

    @property
    def date(self):
        return self._entrie.date

    @property
    def time(self):
        return self._entrie.time

    def run_postprocessing(self):
        return super().run_postprocessing()


class TriggerObject(PostProcessing):
    def __init__(self, database_url: str, tablename: str, filter_dict: dict = {}):
        super().__init__(database_url=database_url)
        self.tablename = tablename
        self.filter = filter_dict
        self.model = TriggerPacketDB.get_model(tablename=self.tablename)
        self._entrie = self.session.query(self.model).filter_by(**self.filter).one()

        self.test_start_time = None

    def run_postprocessing(self):
        self.data_integrity()
        return super().run_postprocessing()

    def data_integrity(self):
        # Check if the data is complete and consistent
        # check data type and do conversion if needed
        if type(self.data_post.time.values[0]) == str:
            # Conversion to datetime.time type
            self.logger.info("Converting time data to datetime")
            # Assuming the time is in 'HH:MM:SS.micros' format
            time_series = self.data_post.time.apply(
                lambda x: datetime.datetime.strptime(x, "%H:%M:%S.%f").time()
            )
            self.data_post.time = time_series

    def update_attributes_from_test(self, attributes: dict):
        self.__dict__.update(attributes)

    @property
    def data(self):
        df = self.get_dataframe(self.tablename, self.filter)
        return df[[c for c in df.columns]]

    @property
    def test_id(self):
        return self._entrie.test_id

    @property
    def timestamp(self):
        return self._entrie.timestamp

    @property
    def date(self):
        return self._entrie.date

    @property
    def time(self):
        return self._entrie.time

    @property
    def state(self):
        return self._entrie.state

    @property
    def checked_time(self):
        return self.time

    @property
    def checked_time_ms(self):
        return self.time_to_milliseconds(self.checked_time)

    @staticmethod
    def time_to_milliseconds(t):
        return (t.hour * 3600 + t.minute * 60 + t.second) * 1000 + t.microsecond // 1000

    def get_delta_checked_time(self):
        if self.test_start_time is None:
            raise ValueError("test_start_time is not set")
        return PostProcessingTest.compute_dt(self.test_start_time, self.checked_time)


class PostProcessingTest(PostProcessing):
    def __init__(self, database_url: str, test_id: str):
        super().__init__(database_url=database_url)
        self.test_id = test_id
        self.filter_dict = {"test_id": self.test_id}
        self.data = {}
        self.setup_logger()

        self.class_map = {
            "IMU": IMUObject,
            "TESTS": TestObject,
            "TRIGGER": TriggerObject,
        }
        self.imus = {}
        self.triggers = {}

        self._postprocessed = False

    def parse_test_id(self):
        test_id_parts = self.test_id.split("_")
        if len(test_id_parts) != 4:
            raise ValueError("Invalid test ID format")
        return test_id_parts[0], test_id_parts[1], test_id_parts[2], test_id_parts[3]

    def __repr__(self):
        test_prefix, visit, ratio, run = self.parse_test_id()
        return f"{self.__class__.__name__} - test_prefix= {test_prefix}, visit= {visit},  ratio= {ratio}, run= {run}"

    def is_completed(self):
        status = [
            s[0]
            for s in self.session.query(TestDB.status)
            .filter_by(**self.filter_dict)
            .all()
        ][0]
        if status == "completed":
            return True
        else:
            return False

    def run_postprocessing(self):
        success = self.seek_tables()
        completed = self.is_completed()
        if success and completed:
            self.logger.info(
                f"Postprocessing ready to continue -- TEST_ID= {self.test_id} -- STARTING POSTPROCESS"
            )
            self.create_imu_cluster()
            self.create_trigger_cluster()

            # update startime attribute of imus and trigger sensor
            self.update_clusters_attributes()

            # Starting data segmentation
            [imu.run_postprocessing() for imu in self.imus.values()]

            self.data["sway_diffs"] = {}
            [
                self.data["sway_diffs"].update({name: imu.sway_diff})
                for name, imu in self.imus.items()
            ]

            self._postprocessed = True
            # self.dump_to_database()

        else:
            self.logger.error("Postprocessing not ready to continue")

    def seek_tables(self) -> bool:
        needed_tables = list(self.class_map.keys())
        seleted_tables = [
            t for nt in needed_tables for t in self.tables if t.startswith(nt)
        ]
        for t in seleted_tables:
            df = self.get_dataframe(tablename=t, filter_dict=self.filter_dict)
            if len(df) > 0:
                prefix = t.split("_")[0]
                if prefix in self.class_map:
                    obj_class = self.class_map[prefix]
                    obj = obj_class.init_from_engine(
                        engine=self.engine,
                        tablename=t,
                        filter_dict=self.filter_dict,
                    )
                    self.data.update({t: obj})
                    self.logger.info(f"Add {t} data object")
            else:
                self.logger.warning(f"No data found for {t}")

        self.logger.info("Check tables validity")
        return self.check_tables_validity()

    def check_tables_validity(self):
        """
        Checks whether a test is valid.

        A test is considered valid if it contains at least:
        - One table with a name starting with 'IMU_'
        - One table with a name starting with 'TESTS'
        - One table with a name starting with 'TRIGGER'

        Returns:
            bool: True if the test is valid, False otherwise.
        """
        a = sum(name.startswith("IMU_") for name in self.data)
        b = sum(name.startswith("TRIGGER") for name in self.data)
        c = sum(name.startswith("TESTS") for name in self.data)

        return a > 0 and b > 0 and c > 0

    def is_valid(self):
        """
        Returns whether the test is valid."
        """
        return self.check_tables_validity()

    def create_imu_cluster(self):
        """
        Creates an IMU cluster for the test.
        """
        self.logger.info("Creating imu cluster")
        [
            self.imus.update({name: self.data[name]})
            for name in self.data
            if name.startswith("IMU_")
        ]

    def create_trigger_cluster(self):
        """
        Creates a trigger cluster for the test.
        """
        self.logger.info("Creating trigger cluster")
        [
            self.triggers.update({name: self.data[name]})
            for name in self.data
            if name.startswith("TRIGGER_")
        ]

    def update_clusters_attributes(self):
        """
        Updates the clusters with the latest data.
        """
        self.logger.info("Updating clusters")
        self.logger.info("Updating start_time attribute")

        attr2update = {
            "test_start_time": self.start_time,
            "test_end_time": self.end_time,
            "test_first_check_time": self.checked_times[0],
            "test_second_check_time": self.checked_times[1],
        }

        cluster = dict()
        cluster.update(self.imus)
        cluster.update(self.triggers)
        [obj.update_attributes_from_test(attr2update) for obj in cluster.values()]

    def get_time_intervals(self):
        """
        Returns the start and end time of the test.
        """
        start_times = []
        end_times = []
        for _, imu_pp in self.imus.items():
            start_times.append(imu_pp.start_time)
            end_times.append(imu_pp.end_time)
        return min(start_times), max(end_times)

    @property
    def post_processed(self):
        return self._postprocessed

    @property
    def start_time(self):
        """
        Returns the start time of the test.
        return type : datetime.time
        """
        return self.get_time_intervals()[0]

    @property
    def end_time(self):
        """
        Returns the end time of the test.
        return type : datetime.time
        """
        return self.get_time_intervals()[1]

    @property
    def duration(self):
        # FIXME: return type should be datetime.timedelta
        """
        Returns the duration of the test.
        return type : datetime.timedelta
        """

        pass

    @property
    def visit(self):
        return self.test.visit

    @property
    def checked_times(self):
        """
        Returns the checked time of the test.
        return type : datetime.time
        """
        checked_times = [
            trigger.checked_time for name, trigger in self.triggers.items()
        ]
        return min(checked_times), max(checked_times)

    @property
    def date(self):
        """
        Returns the date of the test.
        return type : datetime.date
        """
        return self.data["TESTS"].date

    @staticmethod
    def compute_dt(time_0: datetime.datetime, time_1: datetime.datetime):
        """
        Computes the difference between two times.
        return type : datetime.timedelta
        """
        reference_date = datetime.date.today()
        t0 = datetime.datetime.combine(reference_date, time_0)
        t1 = datetime.datetime.combine(reference_date, time_1)
        return t1 - t0

    @property
    def sway_diffs(self):
        return np.array([v for v in self.data["sway_diffs"].values()])

    @property
    def sway_diff(self):
        if len(self.sway_diffs) == 1:
            return self.sway_diffs[0]
        elif len(self.sway_diffs) > 1:
            # compute mean value if std is less than 10%
            std, mean = np.std(self.sway_diffs), np.mean(self.sway_diffs)
            if std < 0.1 * mean:
                return mean
            else:
                raise ValueError(
                    "Standard deviation is too high compared to the mean. Please check your data."
                )
        else:
            raise ValueError("No sway diffs found. Please check your data.")

    @property
    def imu(self):
        if len(self.imus.keys()) > 1:
            raise ValueError("Multiple IMUs found. Please specify which one to use.")
        return next(iter(self.imus.values()))

    def get_checked_times(self, reference="relative"):
        """
        reference : str, default is "relative", can be either "absolute" or "relative".
        Returns the checked times of all triggers.
        return type : list(datetime.timedelta)"
        """
        if reference == "absolute":
            return self.checked_times
        else:
            return [self.compute_dt(self.start_time, t) for t in self.checked_times]

    @property
    def test(self):
        return self.data["TESTS"]

    def dump_to_database(self):
        tablename = f"POST_PROCESSED_TESTS"
        model = PostProcessedTestDB.get_model(
            tablename=tablename, class_name="PostProcessedTestDB"
        )
        model.metadata.create_all(bind=self.engine)
        session = Session(bind=self.engine)
        try:
            entry = [
                model(
                    test_id=self.test_id,
                    sway_diff=self.sway_diff,
                    mean_sway_zone_1=self.imu.mean_sway_zone("zone_1"),
                    max_sway_zone_2=self.imu.max_sway_zone("zone_2"),
                    mean_norm_acc_zone_1=self.imu.mean_norm_acc_zone("zone_1"),
                    mean_norm_acc_zone_2=self.imu.mean_norm_acc_zone("zone_2"),
                    move_id=self.test.move_id,
                    door_width=self.test.door_width,
                    ratio=self.test.ratio,
                    run=self.test.run,
                    visit=self.test.visit,
                )
            ]

            # Ajoute tous les objets d'un coup
            session.add_all(entry)
            session.commit()

        except Exception as e:
            session.rollback()
            print(f"Erreur lors de l'insertion dans la base de données : {e}")

        finally:
            session.close()


class PostProcessingRatio(PostProcessing):
    def __init__(self, database_url: str, test_prefix: str, visit: int, ratio: float):
        super().__init__(database_url=database_url)

        self.test_prefix = test_prefix
        self.ratio = ratio
        self.visit = visit

        self.tests = {}

        self._selected_imu = None

        self.U_ratio = None
        self.U_ratio_interval = (
            None,
            None,
        )  # inferior and superior bounds of the uncertainty ratio

        self.run_postprocessing()

    def run_postprocessing(self):
        self.loads_tests()
        self.check_imu_names()

    def check_imu_names(self):
        if all(x == self.imu_names[0] for x in self.imu_names):
            self.logger.info(f"Same IMU used for every tests")
            self.logger.info("Setting {} as seletected_imu".format(self.imu_names[0]))
            self.selected_imu = self.imu_names[0]
        else:
            self.logger.warning(f"Different IMUs used for tests")
            self.logger.warning(
                "Please define selected IMU using self.selected_imu = 'imu_name'"
            )

    @property
    def test_names(self):
        return [f"{self.test_prefix}_{self.visit}_{self.ratio}_{i}" for i in range(3)]

    def loads_tests(self):
        """
        Loads the tests from the database.
        """

        for i in range(3):
            test_id = self.test_names[i]
            self.tests[test_id] = PostProcessingTest.init_from_engine(
                engine=self.engine, test_id=test_id
            )
            self.tests[test_id].run_postprocessing()

    @property
    def mean_sway_diff(self):
        return np.mean(
            [test.sway_diff for test in self.tests.values() if test.is_completed()]
        )

    @property
    def selected_imu(self):
        if self._selected_imu is None:
            raise ValueError("No IMU selected.")
        else:
            return self._selected_imu

    @selected_imu.setter
    def selected_imu(self, imu_name: str):
        self._selected_imu = imu_name

    def mean_sway_zone(self, zone: str):
        """
        zone :str can be 'zone_1' or 'zone_2'
        """

        return np.array(
            [
                test.imu.mean_sway_zone(zone)
                for test in self.tests.values()
                if test.post_processed
            ]
        ).mean()

    def mean_max_sway_zone(self, zone: str):
        """
        zone :str can be 'zone_1' or 'zone_2'
        """

        return np.array(
            [
                test.imu.max_sway_zone(zone)
                for test in self.tests.values()
                if test.post_processed
            ]
        ).mean()

    @property
    def mean_sway_zone_1(self):
        return self.mean_sway_zone("zone_1")

    @property
    def mean_sway_zone_2(self):
        return self.mean_sway_zone("zone_2")

    def sway_angles(self):
        d = []
        for test in self.tests.values():
            d.append(test.imu.compute_zones_sway())

        df = pd.DataFrame(d)
        df.index.name = "run"

        return df

    def mean_max_norm_acc_zone(self, zone: str):
        """
        zone :str can be 'zone_1' or 'zone_2'
        """

        return np.array(
            [
                test.imu.max_norm_acc_zone(zone)
                for test in self.tests.values()
                if test.post_processed
            ]
        ).mean()

    def mean_norm_acc_zone(self, zone: str):
        """
        zone :str can be 'zone_1' or 'zone_2'
        """

        return np.array(
            [
                test.imu.mean_norm_acc_zone(zone)
                for test in self.tests.values()
                if test.post_processed
            ]
        ).mean()

    @property
    def imu_names(self):
        return [imu_name for test in self.tests.values() for imu_name in test.imus]

    def __getitem__(self, run: int) -> PostProcessingTest:
        return self.get_test_by_run(run)

    def get_test_by_run(self, run: int) -> PostProcessingTest:
        test_id = self.test_names[run]
        return self.tests[test_id]

    def __repr__(self):
        return f"{self.__class__.__name__} - test_prefix= {self.test_prefix} - visit= {self.visit} - ratio= {self.ratio}"

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Gives the data from the sway angles in degrees for the current test for both zones.
        """
        return self.sway_angles()

    def plot_zones_yaw_inflexions(self, savefig=False):
        """
        Plots the yaw inflexions for both zones.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12 * 1.5, 4 * 1.5))
        fig.suptitle(
            "Yaw angles overs time and inflexion points - Ratio: {:.2f}".format(
                self.ratio
            )
        )

        ax1, ax2 = axs
        ax1.set_title(f"zone_1")
        ax2.set_title(f"zone_2")
        Nt = len(self.test_names)
        for i in range(Nt):
            test_id = self.test_names[i]
            imu = self.tests[test_id].imu
            for zone in imu.data_post.zone.unique():
                if zone == "zone_1":
                    ax = ax1
                    df = pd.DataFrame(
                        self.dataframe[zone].values, columns=["mean_diff_ang"]
                    )
                else:
                    ax = ax2
                    df = pd.DataFrame(
                        self.dataframe[zone].values, columns=["max_diff_ang"]
                    )
                data_zone = imu.data_post[imu.data_post.zone == zone]
                yaw_y = data_zone.yaw.values
                yaw_x = data_zone.delta_time.values
                yaw_inflextion_y = data_zone[
                    data_zone.inflexion_point == True
                ].yaw.values
                yaw_inflextion_x = data_zone[
                    data_zone.inflexion_point == True
                ].delta_time.values
                ax.plot(yaw_x, yaw_y, ".-", label=f"yaw run={i}")
                ax.plot(
                    yaw_inflextion_x, yaw_inflextion_y, "o", label=f"inf pnts run={i}"
                )

                df.index.name = "run"

                textstr = (
                    df.to_string()
                    + "\n mean: "
                    + str(round(self.dataframe[zone].mean(), 2))
                    + "\n std: "
                    + str(round(self.dataframe[zone].std(), 2))
                )
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                ax.text(
                    0.75,
                    0.25,
                    textstr,
                    transform=ax.transAxes,
                    # fontsize=12,
                    verticalalignment="top",
                    bbox=props,
                )

        for ax in axs:
            ax.grid(True)
            ax.legend(loc="upper left")
            ax.set_ylabel("angle [°]")
            ax.set_xlabel("time [s]")
        if not savefig:
            plt.show()
        else:
            plt.savefig(f"{self.test_prefix}_{self.ratio}.png")
            plt.close()

    def compute_uncertainties_interval(self, shoulder_width):
        # shoulder incertainties
        u_Ls = 0.1 / 12**0.5
        U_Ls = (u_Ls / shoulder_width) ** 2
        # door uncertainties
        u_Ld = 0.1 / 12**0.5

        door_width = self.ratio * shoulder_width
        U_Ld = (u_Ld / door_width) ** 2

        u_ratio = abs(self.ratio) * (U_Ls + U_Ld) ** 0.5
        self.U_ratio = 2 * u_ratio
        self.U_ratio_interval = (self.ratio - self.U_ratio, self.ratio + self.U_ratio)


class PostProcessingPatient(PostProcessing):
    def __init__(self, database_url: str, move_id: str):
        super().__init__(database_url=database_url)
        self.move_id = move_id
        self.tests_ratio = []

        self.run_postprocessing()

    def run_postprocessing(self):
        self.get_all_tests()

    def get_patient(self):
        patients = Patients(engine=self.engine)
        patient = patients[self.move_id]
        return patient

    @property
    def patient(self):
        """
        Return an instance of Patient associated with the current move ID.
        See src/record/core/test.py --> class PatientDB for specific attributes
        """
        return self.get_patient()

    @property
    def shoulder_width(self):
        return self.patient.shoulder_width

    def get_all_tests(self):
        tablename = "TESTS"
        # table = Table(tablename, self.metadata, autoload=True)
        model = TestDB.get_model(tablename)
        tests_ratio = (
            self.session.query(model)
            .filter_by(move_id=self.move_id, visit=0, run=0)
            .all()
        )
        self.tests_ratio = []
        for test in tests_ratio:
            for i in range(2):
                self.tests_ratio.append(
                    PostProcessingRatio.init_from_engine(
                        engine=self.engine,
                        test_prefix=self.move_id,
                        visit=i,
                        ratio=test.ratio,
                    )
                )

    def get_data(self):
        Ntr = len(self.tests_ratio)
        data = np.zeros((Ntr, 4))
        for i, test in enumerate(self.tests_ratio):
            data[i, 0] = test.ratio
            data[i, 1] = test.mean_sway_diff
            data[i, 2] = test.visit
            data[i, 3] = test.U_ratio

        columns = ["Ratio", "Mean Sway Diff", "Visit", "err_U_ratio"]
        df = pd.DataFrame(data, columns=columns)
        return df

    # def __getitem__(self, test_name):
    #     return self.get_test_by_name(test_name)
    def __getitem__(self, ratio):
        return self.get_tests_by_ratio(ratio)

    def get_test_by_name(self, name):
        test_prefix, visit, ratio, run = name.split("_")
        ratio = float(ratio)
        run = int(run)
        for test in self.tests_ratio:
            if (
                test.test_prefix == test_prefix
                and test.visit == visit
                and test.ratio == float(ratio)
            ):
                return test.tests[name]
        raise ValueError(f"Test {name} not found")

    def get_tests_by_ratio(self, ratio):
        return [test for test in self.tests_ratio if test.ratio == float(ratio)]

    def plot_final(self, savefig=False):
        final = self.get_data()
        plt.figure()
        df0 = final[final["Visit"] == 0]
        df1 = final[final["Visit"] == 1]
        plt.plot(
            df0["Ratio"].values, df0["Mean Sway Diff"].values, "bo", label="Visit 0"
        )
        plt.plot(
            df1["Ratio"].values, df1["Mean Sway Diff"].values, "ro", label="Visit 1"
        )
        # draw horizontal line at y=0
        plt.axhline(y=0, color="r", linestyle="-.")
        plt.grid()
        plt.legend()
        # x ticks every 0.1 from 0.8 to 2.1
        plt.xticks(np.linspace(0.8, 2.1, 14))
        plt.yticks(np.arange(-5, 95, 10))
        plt.xlabel("Ratio")
        plt.ylabel("Angular Sway Difference [°]")
        plt.title(f"{self.move_id} -Angular Sway Difference vs Ratio")
        if not savefig:
            plt.show()
        else:
            plt.savefig(f"{self.move_id}_plot_final.png")
            plt.close()

    def export_plots(self):
        self.plot_final(savefig=True)
        [rtest.plot_zones_yaw_inflexions(savefig=True) for rtest in self.tests_ratio]

    def compute_uncertainties(self):
        for test in self.tests_ratio:
            test.compute_uncertainties_interval(self.patient.shoulder_width)

    @property
    def tests_names(self):
        return [name for tt in self.tests_ratio for name in tt.test_names]

    def _tests(self):
        for tt in self.tests_ratio:
            tests = tt.tests
            for key, val in tests.items():
                yield val

    @property
    def tests(self):
        return list(self._tests())

    def postprocess_tests(self):
        # Colonnes attendues
        columns = [
            "test_id",
            "sway_diff",
            "mean_sway_zone_1",
            "max_sway_zone_2",
            "mean_norm_acc_zone_1",
            "mean_norm_acc_zone_2",
            "order",
        ]

        # DataFrame vide initial

        # Charger les données depuis la base
        tablename = "TESTS"
        model = TestDB.get_model(tablename)

        df_cluster = []
        # Lire les données
        for i in range(2):
            df = pd.read_sql(
                self.session.query(model)
                .filter_by(move_id=self.move_id, visit=i)
                .statement,
                self.session.bind,
            )

            # Trier et réindexer
            df = df.sort_values(by=["start_time"]).reset_index(drop=True)

            # Ajouter test_id et ordre dans dd
            dd = pd.DataFrame(columns=columns)
            dd["test_id"] = df["test_id"]
            dd["order"] = df.index

            # Remplir les colonnes depuis self.tests
            for test in self.tests:
                if test.post_processed:
                    # Met à jour les lignes correspondantes dans dd
                    dd.loc[dd["test_id"] == test.test_id, "sway_diff"] = test.sway_diff
                    dd.loc[
                        dd["test_id"] == test.test_id, "mean_sway_zone_1"
                    ] = test.imu.mean_sway_zone("zone_1")
                    dd.loc[
                        dd["test_id"] == test.test_id, "mean_sway_zone_2"
                    ] = test.imu.mean_sway_zone("zone_2")
                    dd.loc[
                        dd["test_id"] == test.test_id, "max_sway_zone_1"
                    ] = test.imu.max_sway_zone("zone_1")
                    dd.loc[
                        dd["test_id"] == test.test_id, "max_sway_zone_2"
                    ] = test.imu.max_sway_zone("zone_2")
                    dd.loc[
                        dd["test_id"] == test.test_id, "mean_norm_acc_zone_1"
                    ] = test.imu.mean_norm_acc_zone("zone_1")
                    dd.loc[
                        dd["test_id"] == test.test_id, "mean_norm_acc_zone_2"
                    ] = test.imu.mean_norm_acc_zone("zone_2")
                    dd.loc[
                        dd["test_id"] == test.test_id, "max_norm_acc_zone_1"
                    ] = test.imu.max_norm_acc_zone("zone_1")
                    dd.loc[
                        dd["test_id"] == test.test_id, "max_norm_acc_zone_2"
                    ] = test.imu.max_norm_acc_zone("zone_2")

            df_cluster.append(dd)

        self.data_test_post_processed = pd.concat(df_cluster)

    def dump_postprocessed_tests(self):
        self.postprocess_tests()
        tablename = f"POST_PROCESSED_TESTS"
        model = PostProcessedTestDB.get_model(
            tablename=tablename, class_name="PostProcessedTestDB"
        )
        model.metadata.create_all(bind=self.engine)
        session = Session(bind=self.engine)
        try:
            entry = [
                model(
                    test_id=row["test_id"],
                    sway_diff=row["sway_diff"],
                    mean_sway_zone_1=row["mean_sway_zone_1"],
                    max_sway_zone_2=row["max_sway_zone_2"],
                    mean_norm_acc_zone_1=row["mean_norm_acc_zone_1"],
                    mean_norm_acc_zone_2=row["mean_norm_acc_zone_2"],
                    order=row["order"],
                )
                for _, row in self.data_test_post_processed.iterrows()
            ]

            # Ajoute tous les objets d'un coup
            session.add_all(entry)
            session.commit()

        except Exception as e:
            session.rollback()
            print(f"Erreur lors de l'insertion dans la base de données : {e}")

        finally:
            session.close()

    def get_data_raw_dataframe(self):
        self.postprocess_tests()
        columns = [
            "move_id",
            "session",
            "shoulder_width",
            "ratio",
            "zone",
            "trial",
            "trial_rank",
            "mean_angle",
            "max_angle",
            "acceleration",
        ]

        Ntests = len(self.tests)
        Nratio = 12
        Nession = 2
        Nzone = 2
        Ntrial = 3
        Nrow = Nratio * Nession * Nzone * Ntrial

        # Initialise le DataFrame vide avec Nrow lignes
        df = pd.DataFrame(index=range(Nrow), columns=columns)
        df["move_id"] = self.move_id  # Même valeur pour toutes les lignes

        row_idx = 0  # Index global pour remplir les lignes du DataFrame

        for i in range(Ntests):
            test = self.tests[i]
            session = test.visit
            ratio = test.test.ratio
            trial = test.test.run

            # Récupération du rang (ordre) de ce test dans les données post-traitées
            df_test_post = self.data_test_post_processed[
                self.data_test_post_processed["test_id"] == test.test_id
            ]

            if df_test_post.empty:
                continue  # saute ce test si pas de données

            trial_rank = int(df_test_post["order"].values[0])

            for j in range(Nzone):
                zone = "Z" + str(j + 1)

                try:
                    mean_angle = df_test_post[f"mean_sway_zone_{j + 1}"].values[0]
                    max_angle = df_test_post[f"max_sway_zone_{j + 1}"].values[0]
                    acceleration = df_test_post[f"mean_norm_acc_zone_{j + 1}"].values[0]
                except KeyError:
                    # au cas où la zone n'existe pas dans les colonnes
                    mean_angle = max_angle = acceleration = None

                # Remplir une ligne du DataFrame
                df.loc[row_idx] = {
                    "move_id": self.move_id,
                    "session": session,
                    "shoulder_width": self.shoulder_width
                    if hasattr(self, "shoulder_width")
                    else None,
                    "ratio": ratio,
                    "zone": zone,
                    "trial": trial,
                    "trial_rank": trial_rank,
                    "mean_angle": mean_angle,
                    "max_angle": max_angle,
                    "acceleration": acceleration,
                }

                row_idx += 1

        return df

    def get_data_mean_dataframe(self):
        columns = [
            "move_id",
            "session",
            "shoulder_width",
            "ratio",
            "zone",
            "mean_angle",
            "max_angle",
            "mean_acceleration",
            "max_acceleration",
        ]

        Nratio = len(self.tests_ratio)
        Nession = 2
        Nzone = 2
        Nrow = Nratio * Nzone

        # Initialise le DataFrame vide avec Nrow lignes
        df = pd.DataFrame(index=range(Nrow), columns=columns)
        df["move_id"] = self.move_id  # Même valeur pour toutes les lignes
        row_idx = 0  # Index global pour remplir les lignes du DataFrame

        for i in range(Nratio):
            ratio_test = self.tests_ratio[i]
            session = ratio_test.visit
            ratio = ratio_test.ratio

            for j in range(Nzone):
                zone = "Z" + str(j + 1)
                try:
                    mean_angle = ratio_test.mean_sway_zone(f"zone_{j+1}")
                    max_angle = ratio_test.mean_max_sway_zone(f"zone_{j+1}")
                    mean_acceleration = ratio_test.mean_norm_acc_zone(f"zone_{j+1}")
                    max_acceleration = ratio_test.mean_max_norm_acc_zone(f"zone_{j+1}")

                    df.loc[row_idx] = {
                        "move_id": self.move_id,
                        "session": session,
                        "shoulder_width": self.shoulder_width
                        if hasattr(self, "shoulder_width")
                        else None,
                        "ratio": ratio,
                        "zone": zone,
                        "mean_angle": mean_angle,
                        "max_angle": max_angle,
                        "mean_acceleration": mean_acceleration,
                        "max_acceleration": max_acceleration,
                    }
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"test prefix ={ratio_test.test_prefix}")
                    print(f"test visit = {ratio_test.visit}")
                    print(f"test ratio = {ratio_test.ratio}")
                    input("Press Enter to continue...")

                row_idx += 1

        return df

    def get_data_post_dataframe(self):
        self.compute_uncertainties()
        columns = [
            "move_id",
            "session",
            "shoulder_width",
            "ratio",
            "delta_Z1_Z2",
            "uncertainty_ratio",
        ]

        data_in = self.get_data()
        df = pd.DataFrame(index=range(len(data_in)), columns=columns)
        # Iterate through the data dataframe and populate df
        df["move_id"] = self.move_id
        df["shoulder_width"] = self.shoulder_width
        df["session"] = data_in["Visit"]
        df["delta_Z1_Z2"] = data_in["Mean Sway Diff"]
        df["ratio"] = data_in["Ratio"]
        df["uncertainty_ratio"] = data_in["err_U_ratio"]

        return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from record.core.constants import MODULE_PATH

    wdir = "/".join(MODULE_PATH.split("/")[:-2]) + "/scripts/"
    filename = "all_tries_before_20-06-2025/database.db"
    # database_url = (
    #     "sqlite://///Users/celmo/Git/record-imucap/src/record/database/database.db"
    # )
    # pp = PostProcessingPatient(database_url=database_url, move_id="M01IV")
    # # pp["M01IV_0.9_0"].imus["IMU_BNO055_01"].plot_zones_yaw_inflexions()
    # # final = pp.get_data()
    # # plt.figure()
    # # plt.plot(final["Ratio"].values, final["Mean Swing Diff"].values, "bo")
    # # plt.show()

    database_url = wdir + filename
    patients = ["A09AC", "E06BN", "L11NM", "N02OL", "U08VA"]
    cluster_data_raw = []
    cluster_data_mean = []
    cluster_data_post = []
    for patient in patients:
        pp = PostProcessingPatient(database_url=database_url, move_id=patient)
        cluster_data_raw.append(pp.get_data_raw_dataframe())
        cluster_data_mean.append(pp.get_data_mean_dataframe())
        cluster_data_post.append(pp.get_data_post_dataframe())

    data_raw = pd.concat(cluster_data_raw)
    data_mean = pd.concat(cluster_data_mean)
    data_post = pd.concat(cluster_data_post)

    writer = pd.ExcelWriter(wdir + "data.xlsx", engine="xlsxwriter")
    data = {"data_raw": data_raw, "data_mean": data_mean, "data_post": data_post}
    for key, val in data.items():
        val.to_excel(writer, sheet_name=key, index=False)
    writer.close()

    # Export to xlsx
    # data_raw.to_excel(wdir + f"data_raw.xlsx", index=False)
    # data_mean.to_excel(wdir + f"data_mean.xlsx", index=False)
    # data_post.to_excel(wdir + f"data_post.xlsx", index=False)

    # # Récupération du rang (ordre) de ce test dans les données post-traitées
    # df_test_post = pp.data_test_post_processed[
    #     pp.data_test_post_processed["test_id"] == test.test_id
    # ]

    # if df_test_post.empty:
    #     continue  # saute ce test si pas de données

    # for j in range(Nzone):
    #     zone = "Z" + str(j + 1)

    #     try:
    #         mean_angle = df_test_post[f"mean_sway_zone_{j + 1}"].values[0]
    #         max_angle = df_test_post[f"max_sway_zone_{j + 1}"].values[0]
    #         acceleration = df_test_post[f"mean_norm_acc_zone_{j + 1}"].values[0]
    #     except KeyError:
    #         # au cas où la zone n'existe pas dans les colonnes
    #         mean_angle = max_angle = acceleration = None

    #     # Remplir une ligne du DataFrame
    #     df.loc[row_idx] = {
    #         "move_id": pp.move_id,
    #         "session": session,
    #         "shoulder_width": pp.shoulder_width
    #         if hasattr(pp, "shoulder_width")
    #         else None,
    #         "ratio": ratio,
    #         "zone": zone,
    #         "mean_angle": mean_angle,
    #         "max_angle": max_angle,
    #         "acceleration": acceleration,
    #     }

    #     row_idx += 1

    # ptest = PostProcessingTest(database_url=database_url, test_id="E06BN_0_2.0_0")

    # needed_tables = list(ptest.class_map.keys())
    # seleted_tables = [t for nt in needed_tables for t in ptest.tables if t.startswith(nt)]
    # patients = [
    #             # "A09AC",
    #             # "E06BN"
    #             "L11NM",
    #             # "N02OL",
    #             # "U08VA"
    #             ]
    # for patient in patients:
    #     pp = PostProcessingPatient(database_url=database_url, move_id=patient)
    # #     pp.plot_final(savefig=True)
    # pp.get_test_by_name('E06BN_0_1.0_0').imu.plot_zones_yaw_inflexions()
    # tablename = "TESTS"
    # # table = Table(tablename, self.metadata, autoload=True)
    # model = TestDB.get_model(tablename)
    # tests_ratio = (
    #     pp.session.query(model).filter_by(move_id=pp.move_id, run=0).all()
    # )

    # for test in tests_ratio:
    #     for i in range(2):
    #         pp.tests_ratio.append(
    #             PostProcessingRatio.init_from_engine(
    #                 engine=pp.engine,
    #                 test_prefix=pp.move_id,
    #                 visit=i,
    #                 ratio=test.ratio,
    #             )
    #         )
    # pp.plot_final()
    # pp["M01IV_0.9_0"].imus["IMU_BNO055_01"].plot_zones_yaw_inflexions()
    # final = pp.get_data()
    # plt.figure()
    # plt.plot(final["Ratio"].values, final["Mean Sway Diff"].values, "bo")
    # plt.show()

    # test_processing = PostProcessingTest(database_url=database_url,
    #                                     test_id="E06BN_0_0.9_0")

    # imu_pp = IMUObject(database_url=database_url,tablename="IMU_BNO055_01", filter_dict={"test_id":"E06BN_0_0.9_0"})

    # test_processing.run_postprocessing()
    # test_processing.imu.dump_to_database()
    # test_processing.imus[list(test_processing.imus.keys())[0]].plot_zones_yaw_inflexions(hold=False)

    # plt.show()

    # test_ids = [test_id for test_id in get_test_ids(database_url) if test_id.startswith("A09AC")]
    # for test_id in test_ids:
    #     try:
    #         print("-----------")
    #         print(f"Test ID: {test_id}")
    #         test_processing = PostProcessingTest(database_url=database_url,
    #                                             test_id=test_id)
    #         test_processing.run_postprocessing()
    #         test_processing.imus[list(test_processing.imus.keys())[0]].plot_zones_yaw_inflexions(hold=True)
    #     except Exception as e:
    #         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #         print(f"Error processing test {test_id}:, \n{e}")
    #         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #         input("Press Enter to continue...")

    # plt.show()

    # lc=0.25
    # imu = test_processing.imu
    # imu.data_post["inflexion_point"] = imu.data_post.shape[0] * [False]
    # MIN_INF_LOC_SIZE = {
    #     "zone_1":3,
    #     "zone_2":2,
    # }
    # zone = "zone_2"
    # zone_yaw = imu.data_post[imu.data_post.zone == zone]["yaw"].values.astype(
    #     np.float64
    # )

    # zone_delta_time = imu.data_post[imu.data_post.zone == zone][
    #     "delta_time"
    # ].values.astype(np.float64)

    # y_meas = np.append(zone_yaw, 0.0)
    # x_meas = np.append(zone_delta_time, zone_delta_time[-1] + 0.25)
    # Y = interpolate.interp1d(x_meas, y_meas, kind="nearest")
    # N_interp = 10000
    # x_interp = np.linspace(x_meas.min(), x_meas.max(), num=N_interp)
    # y_interp = Y(x_interp)
    # zone_inf_loc = np.array([])
    # lookahead = int(lc * N_interp)
    # while zone_inf_loc.size < MIN_INF_LOC_SIZE[zone]:
    #     fp = findpeaks(method='peakdetect', lookahead=lookahead, interpolate=4)
    #     results = fp.fit(y_interp)
    #     df=results["df"]
    #     if zone == "zone_1":
    #         df=IMUObject.clean_extremas(df)
    #     zone_inf_loc, zone_inf_val = IMUObject.get_inflexion_points(df)
    #     lookahead /= 2
    #     lookahead = int(lookahead)
    #     if lookahead < 10:
    #         break

    # zone_inf_loc = zone_inf_loc[1:-1].astype(int)

    # # Check if there at leat 0.3s between two consecutive inflexion points
    # time_line = [x_interp[loc] for loc in zone_inf_loc]
    # time_diff = np.diff(time_line)
    # # if np.any(time_diff < 0.3):
