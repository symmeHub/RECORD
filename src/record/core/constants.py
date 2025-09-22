"""
Core constants and configuration paths used across the project.

Notes
-----
- Timezone constants are based on pytz (e.g., Europe/Paris).
- Madgwick-related constants assume IMU/MARG typical gains.
- Module paths resolve to the installed `record` package directory.
"""

import os
import pytz
import record

PARIS_TIME_ZONE = pytz.timezone("Europe/Paris")
#
# core.py
# Acquisition frequency at 52 Hz (Shimmer), used for the Madgwick algorithm
DT = 1 / 52
# Filter gain. Defaults to 0.033 for IMU implementations, or to 0.041 for MARG implementations.
GAIN_MARG = 0.041
GAIN_IMU = 0.033

# imu.py
RUN_MAX_FREQ = 35.0  # Hz
MIN_WAIT_TIME = 1 / RUN_MAX_FREQ

# Get module path from the environment variable if it exists, otherwise use the current directory.
MODULE_PATH = record.__path__[0]
DATABASE_DIR = os.path.join(MODULE_PATH, "database")
CONFIG_DIR = os.path.join(MODULE_PATH, "configs")
