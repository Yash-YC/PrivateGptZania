from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from strictyaml import YAML, load

ROOT = Path(__file__).resolve().parent
CONFIG_FILE_PATH = ROOT / "config.yml"
LOG_DIR = ROOT / "logs"
PDF_DIR = ROOT / "pdf"
VECTOR_STORE_DIR = ROOT / "vector_store"


class BaseConfig(BaseModel):
    """
    Application-level config.
    """
    OPENAI_API_KEY: str
    LLM: str
    QUESTIONS: list
    PROMPT_TEMPLATE: str


class Config(BaseModel):
    """Master config object."""
    base_config: BaseConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        base_config=BaseConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
