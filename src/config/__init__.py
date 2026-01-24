"""Configuration management for OCI Anthropic Gateway."""

import json
import logging
import oci
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.constants import DEFAULT_MAX_TOKENS

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oci-gateway")


class Config:
    """Configuration manager for OCI Anthropic Gateway."""

    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration from file.

        Args:
            config_file: Path to configuration JSON file
        """
        self.config_file = config_file
        self.compartment_id: str = ""
        self.endpoint: str = ""
        self.model_aliases: Dict[str, str] = {}
        self.model_definitions: Dict[str, Dict[str, Any]] = {}
        self.default_model_name: str = ""
        self.default_model_conf: Dict[str, Any] = {}
        self.genai_client: Optional[oci.generative_ai_inference.GenerativeAiInferenceClient] = None

        self._load_config()
        self._init_oci_client()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

            with open(config_path, "r") as f:
                custom_config = json.load(f)

            self.compartment_id = custom_config["compartment_id"]
            self.endpoint = custom_config["endpoint"]
            self.model_aliases = custom_config.get("model_aliases", {})
            self.model_definitions = custom_config.get("model_definitions", {})
            self.default_model_name = custom_config.get("default_model")

            self.default_model_conf = self.model_definitions.get(self.default_model_name)

            if not self.default_model_conf:
                logger.error(f"Default model '{self.default_model_name}' not found in definitions")
                raise KeyError("Invalid default_model")

            # Add compartment_id to default_model_conf for convenience
            if "compartment_id" not in self.default_model_conf:
                self.default_model_conf["compartment_id"] = self.compartment_id

            logger.info(f"Config loaded. Default model: {self.default_model_name}")

        except Exception as e:
            logger.error(f"Configuration initialization failed: {e}")
            raise

    def _init_oci_client(self) -> None:
        """Initialize OCI GenAI client."""
        try:
            sdk_config = oci.config.from_file('~/.oci/config', "DEFAULT")
            self.genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=sdk_config,
                service_endpoint=self.endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)
            )
            logger.info("OCI SDK initialized successfully")
        except Exception as e:
            logger.error(f"SDK configuration loading failed: {e}")
            raise

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.

        Args:
            model_name: Requested model name

        Returns:
            Model configuration dictionary
        """
        # Lookup logic
        target_alias = None
        for alias_key, alias_value in self.model_aliases.items():
            if alias_key in model_name.lower():
                target_alias = alias_value
                break

        if not target_alias and model_name in self.model_definitions:
            target_alias = model_name

        if not target_alias:
            logger.info(f"Using default model config for: {model_name}")
            return self.default_model_conf

        model_conf = self.model_definitions.get(target_alias)
        if model_conf:
            logger.info(f"Mapped '{model_name}' -> '{target_alias}' config")
            return model_conf

        return self.default_model_conf


# Global configuration instance
config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Global Config instance

    Raises:
        RuntimeError: If configuration has not been initialized
    """
    global config
    if config is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")
    return config


def init_config(config_file: str = "config.json") -> Config:
    """Initialize global configuration.

    Args:
        config_file: Path to configuration JSON file

    Returns:
        Initialized Config instance
    """
    global config
    config = Config(config_file)
    return config
