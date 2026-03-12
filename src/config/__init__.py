"""Configuration management for OCI Anthropic Gateway."""

import json
import logging
import os
import oci
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.constants import DEFAULT_MAX_TOKENS
from ..utils.guardrails import GuardrailsConfig, build_guardrails_config

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oci-gateway")


class Config:
    """Configuration manager for OCI Anthropic Gateway."""
    ALLOWED_MODEL_TYPES = {"text", "images", "video", "audio"}

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
        self.debug: bool = False
        self.debug_ui_enabled: bool = False
        self.debug_ui_dump_dir: str = "debug_dumps"
        self.debug_ui_index_db: str = "debug_dumps/index.db"
        self.debug_ui_scan_interval_sec: int = 3
        self.debug_ui_auth_mode: str = "none"
        self.debug_ui_auth_token: str = ""
        self.debug_ui_auth_basic_user: str = ""
        self.debug_ui_auth_basic_password: str = ""
        self.debug_ui_max_detail_bytes: int = 2_000_000
        self.debug_redact_media: bool = True
        self.enable_nl_tool_fallback: bool = False
        self.messages_max_items: int = 200
        self.rate_limit_enabled: bool = False
        self.rate_limit_requests: int = 60
        self.rate_limit_window_sec: int = 60
        self.server_host: str = "127.0.0.1"
        self.server_port: int = 8000
        self.server_log_level: str = "warning"
        self.guardrails: Optional[GuardrailsConfig] = None

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
            self.debug = bool(custom_config.get("debug", False))
            debug_ui = custom_config.get("debug_ui", {})
            self.debug_ui_enabled = bool(debug_ui.get("enabled", False))
            self.debug_ui_dump_dir = str(debug_ui.get("dump_dir", "debug_dumps"))
            self.debug_ui_index_db = str(debug_ui.get("index_db", "debug_dumps/index.db"))
            self.debug_ui_scan_interval_sec = int(debug_ui.get("scan_interval_sec", 3))
            self.debug_ui_auth_mode = str(debug_ui.get("auth_mode", "none")).strip().lower()
            self.debug_ui_auth_token = str(os.getenv("DEBUG_UI_AUTH_TOKEN", "")).strip()
            self.debug_ui_auth_basic_user = str(os.getenv("DEBUG_UI_BASIC_USER", "")).strip()
            self.debug_ui_auth_basic_password = str(os.getenv("DEBUG_UI_BASIC_PASSWORD", "")).strip()
            self.debug_ui_max_detail_bytes = int(debug_ui.get("max_detail_bytes", 2_000_000))
            self.debug_redact_media = bool(custom_config.get("debug_redact_media", True))
            self.enable_nl_tool_fallback = bool(custom_config.get("enable_nl_tool_fallback", False))
            self.messages_max_items = int(custom_config.get("messages_max_items", 200))

            if self.messages_max_items < 1:
                raise ValueError("Invalid messages_max_items in config.json (must be >=1)")

            rate_limit = custom_config.get("rate_limit", {})
            self.rate_limit_enabled = bool(rate_limit.get("enabled", False))
            self.rate_limit_requests = int(rate_limit.get("requests", 60))
            self.rate_limit_window_sec = int(rate_limit.get("window_sec", 60))
            if self.rate_limit_requests < 1:
                raise ValueError("Invalid rate_limit.requests in config.json (must be >=1)")
            if self.rate_limit_window_sec < 1:
                raise ValueError("Invalid rate_limit.window_sec in config.json (must be >=1)")

            allowed_auth_modes = {"none", "bearer", "basic"}
            if self.debug_ui_auth_mode not in allowed_auth_modes:
                raise ValueError("Invalid debug_ui.auth_mode in config.json (allowed: none,bearer,basic)")

            server_conf = custom_config.get("server", {})
            host = server_conf.get("host", "127.0.0.1")
            port = server_conf.get("port", 8000)
            log_level = server_conf.get("log_level", "warning")

            if not isinstance(host, str) or not host.strip():
                raise ValueError("Invalid server.host in config.json")

            try:
                port_int = int(port)
            except Exception as e:
                raise ValueError("Invalid server.port in config.json") from e

            if port_int < 1 or port_int > 65535:
                raise ValueError("Invalid server.port in config.json (must be 1-65535)")

            if not isinstance(log_level, str) or not log_level.strip():
                raise ValueError("Invalid server.log_level in config.json")

            normalized_log_level = log_level.strip().lower()
            allowed_log_levels = {"critical", "error", "warning", "info", "debug", "trace"}
            if normalized_log_level not in allowed_log_levels:
                raise ValueError(
                    "Invalid server.log_level in config.json (allowed: critical,error,warning,info,debug,trace)"
                )

            self.server_host = host.strip()
            self.server_port = port_int
            self.server_log_level = normalized_log_level
            self.guardrails = build_guardrails_config(
                custom_config.get("guardrails", {}),
                config_file_path=str(config_path),
            )

            self.default_model_conf = self.model_definitions.get(self.default_model_name)

            if not self.default_model_conf:
                logger.error(f"Default model '{self.default_model_name}' not found in definitions")
                raise KeyError("Invalid default_model")

            # Normalize default model config so call sites can always rely on
            # compartment_id and common defaults being present.
            self.default_model_conf = self._normalize_model_conf(self.default_model_conf, self.default_model_name)

            # Validate model definitions at startup to surface configuration issues early.
            for model_name, model_conf in self.model_definitions.items():
                normalized = self._normalize_model_conf(model_conf, model_name)
                # Persist normalized config to keep runtime behavior consistent.
                self.model_definitions[model_name] = normalized
                if not normalized.get("compartment_id"):
                    logger.warning(
                        "Model '%s' has no compartment_id (and no global fallback). "
                        "Requests to this model will fail.",
                        model_name,
                    )
                if not normalized.get("ocid"):
                    logger.warning(
                        "Model '%s' has empty ocid. Requests to this model will fail.",
                        model_name,
                    )

            logger.info(
                f"Config loaded. Default model: {self.default_model_name} (debug={self.debug}) "
                f"debug_ui={self.debug_ui_enabled} "
                f"rate_limit={self.rate_limit_enabled} "
                f"nl_fallback={self.enable_nl_tool_fallback} "
                f"server={self.server_host}:{self.server_port} log_level={self.server_log_level} "
                f"guardrails={bool(self.guardrails and self.guardrails.enabled)}"
            )

        except Exception as e:
            logger.error(f"Configuration initialization failed: {e}")
            raise

    def _init_oci_client(self) -> None:
        """Initialize OCI GenAI client with multiple authentication methods.

        Authentication priority:
        1. OKE Workload Identity signer
        2. OCI Resource Principal signer
        3. ~/.oci/config API Key (local fallback)
        """
        try:
            signer = None

            try:
                logger.info("Trying OKE Workload Identity authentication")
                signer = oci.auth.signers.get_oke_workload_identity_resource_principal_signer()
                logger.info("Using OKE Workload Identity authentication")
            except Exception as oke_e:
                logger.info(f"OKE Workload Identity unavailable: {oke_e}")

            if signer is None:
                rp_version = os.environ.get("OCI_RESOURCE_PRINCIPAL_VERSION", "")
                if rp_version:
                    logger.info("Using OCI Resource Principal authentication")
                    signer = oci.auth.signers.get_resource_principals_signer()

            if signer is not None:
                self.genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config={},
                    signer=signer,
                    service_endpoint=self.endpoint,
                    retry_strategy=oci.retry.NoneRetryStrategy(),
                    timeout=(10, 360)
                )
            else:
                logger.info("Using OCI API Key authentication from ~/.oci/config")
                sdk_config = oci.config.from_file('~/.oci/config', "DEFAULT")
                self.genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                    config=sdk_config,
                    service_endpoint=self.endpoint,
                    retry_strategy=oci.retry.NoneRetryStrategy(),
                    timeout=(10, 360)
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
            return dict(self.default_model_conf)

        model_conf = self.model_definitions.get(target_alias)
        if model_conf:
            logger.info(f"Mapped '{model_name}' -> '{target_alias}' config")
            return dict(model_conf)

        return dict(self.default_model_conf)

    @classmethod
    def _normalize_model_types(cls, model_types: Any, model_name: str) -> List[str]:
        """Normalize and validate model_types for one model config."""
        if model_types is None:
            return ["text"]

        if not isinstance(model_types, list):
            raise ValueError(f"Invalid model_types for model '{model_name}' (must be an array)")

        normalized_types: List[str] = []
        seen = set()
        for i, item in enumerate(model_types):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Invalid model_types[{i}] for model '{model_name}' (must be a non-empty string)"
                )
            t = item.strip().lower()
            if t not in cls.ALLOWED_MODEL_TYPES:
                allowed = ",".join(sorted(cls.ALLOWED_MODEL_TYPES))
                raise ValueError(
                    f"Invalid model_types value '{item}' for model '{model_name}' (allowed: {allowed})"
                )
            if t not in seen:
                seen.add(t)
                normalized_types.append(t)

        if not normalized_types:
            raise ValueError(f"Invalid model_types for model '{model_name}' (must not be empty)")

        return normalized_types

    def _normalize_model_conf(self, model_conf: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
        """Return a normalized model config with global fallbacks applied."""
        normalized = dict(model_conf or {})
        if not normalized.get("compartment_id"):
            normalized["compartment_id"] = self.compartment_id
        if not normalized.get("max_tokens_key"):
            normalized["max_tokens_key"] = "max_tokens"
        normalized["model_types"] = self._normalize_model_types(normalized.get("model_types"), model_name or "unknown")
        return normalized


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
