"""
Utility helpers for colored messages and YAML configuration IO.
"""

import yaml
from colored import Fore, Back, Style


# Color bindngs enum
class Color(Fore):
    """Semantic color aliases for console output."""

    INFO = Fore.GREEN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    DEBUG = Fore.MAGENTA
    CRITICAL = Fore.RED
    SUCCESS = Fore.GREEN
    RESET = Style.reset


class Message:
    """Colorized message helpers for consistent logs/prints."""

    @classmethod
    def info(cls, message: str):
        """Return a green information string."""
        return f"{Fore.GREEN}{message}{Style.reset}"

    @classmethod
    def warning(cls, message: str):
        """Return a yellow warning string."""
        return f"{Fore.YELLOW}{message}{Style.reset}"

    @classmethod
    def error(cls, message: str):
        """Return a red error string."""
        return f"{Fore.RED}{message}{Style.reset}"

    @classmethod
    def debug(cls, message: str):
        """Return a magenta debug string."""
        return f"{Fore.MAGENTA}{message}{Style.reset}"


def load_config_yml(filename: str) -> dict:
    """
    Load a YAML configuration file and return it as a dict.
    """
    with open(f"{filename}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_config_yml(filename: str, config: dict) -> None:
    """
    Save a Python dict to a YAML configuration file.
    """
    with open(f"{filename}", "w") as f:
        yaml.dump(config, f)
    return None
