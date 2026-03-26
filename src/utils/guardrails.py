"""Guardrails helpers for OCI Anthropic Gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import oci

logger = logging.getLogger("oci-gateway")

DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_BLOCK_MESSAGE = "Request blocked by gateway guardrails policy."


class GuardrailsSdkCompatibilityError(RuntimeError):
    """Raised when the installed OCI SDK lacks required Guardrails types."""


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value).strip() or default


def _as_list_of_str(value: Any, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        raise ValueError("Expected an array of strings")
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _validate_threshold(value: Any, field_name: str) -> float:
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Invalid guardrails.{field_name} in config.json (must be between 0.0 and 1.0)")
    return threshold


@dataclass
class OciContentModerationConfig:
    enabled: bool = True
    categories: List[str] = field(default_factory=lambda: ["OVERALL"])
    threshold: float = DEFAULT_SCORE_THRESHOLD


@dataclass
class OciPromptInjectionConfig:
    enabled: bool = True
    threshold: float = DEFAULT_SCORE_THRESHOLD


@dataclass
class OciPIIDetectionConfig:
    enabled: bool = False
    types: List[str] = field(default_factory=list)


@dataclass
class LocalBlocklistConfig:
    enabled: bool = False
    file: str = "blocklist.txt"
    resolved_path: Optional[Path] = None
    entries: Set[str] = field(default_factory=set)


@dataclass
class GatewayPIIRewriteConfig:
    enabled: bool = False
    action: str = "redact"
    placeholder: str = "[REDACTED]"


@dataclass
class OciNativeInputGuardrailsConfig:
    enabled: bool = True
    content_moderation: OciContentModerationConfig = field(default_factory=OciContentModerationConfig)
    prompt_injection: OciPromptInjectionConfig = field(default_factory=OciPromptInjectionConfig)
    pii_detection: OciPIIDetectionConfig = field(default_factory=OciPIIDetectionConfig)


@dataclass
class OciNativeOutputGuardrailsConfig:
    enabled: bool = False
    content_moderation: OciContentModerationConfig = field(default_factory=OciContentModerationConfig)
    pii_detection: OciPIIDetectionConfig = field(default_factory=OciPIIDetectionConfig)


@dataclass
class OciNativeGuardrailsConfig:
    default_language: str = "en"
    input: OciNativeInputGuardrailsConfig = field(default_factory=OciNativeInputGuardrailsConfig)
    output: OciNativeOutputGuardrailsConfig = field(default_factory=OciNativeOutputGuardrailsConfig)


@dataclass
class GatewayGuardrailsPolicyConfig:
    mode: str = "block"
    block_http_status: int = 400
    block_message: str = DEFAULT_BLOCK_MESSAGE
    log_details: bool = False
    redact_logs: bool = True
    input_failure_mode: str = "closed"
    output_failure_mode: str = "open"


@dataclass
class GatewayInputExtensionsConfig:
    include_system: bool = False
    include_tool_results: bool = True
    local_blocklist: LocalBlocklistConfig = field(default_factory=LocalBlocklistConfig)


@dataclass
class GatewayOutputExtensionsConfig:
    pii_rewrite: GatewayPIIRewriteConfig = field(default_factory=GatewayPIIRewriteConfig)


@dataclass
class GatewayStreamingExtensionsConfig:
    when_output_guardrails_enabled: str = "reject"


@dataclass
class GatewayGuardrailsExtensionsConfig:
    input: GatewayInputExtensionsConfig = field(default_factory=GatewayInputExtensionsConfig)
    output: GatewayOutputExtensionsConfig = field(default_factory=GatewayOutputExtensionsConfig)
    streaming: GatewayStreamingExtensionsConfig = field(default_factory=GatewayStreamingExtensionsConfig)


@dataclass
class ContentModerationFinding:
    categories: List[Dict[str, Any]] = field(default_factory=list)
    triggered: bool = False


@dataclass
class PromptInjectionFinding:
    score: float = 0.0
    triggered: bool = False


@dataclass
class PIIFinding:
    entities: List[Dict[str, Any]] = field(default_factory=list)
    detected: bool = False


@dataclass
class GuardrailsCheckResult:
    passed: bool = True
    issue_detected: bool = False
    blocked_reason: Optional[str] = None
    redacted_content: Optional[str] = None
    content_moderation: Optional[ContentModerationFinding] = None
    prompt_injection: Optional[PromptInjectionFinding] = None
    pii: Optional[PIIFinding] = None
    local_blocklist_matches: List[str] = field(default_factory=list)


@dataclass
class GuardrailsConfig:
    enabled: bool = False
    config_dir: Path = Path("guardrails")
    oci_native: OciNativeGuardrailsConfig = field(default_factory=OciNativeGuardrailsConfig)
    gateway_policy: GatewayGuardrailsPolicyConfig = field(default_factory=GatewayGuardrailsPolicyConfig)
    gateway_extensions: GatewayGuardrailsExtensionsConfig = field(default_factory=GatewayGuardrailsExtensionsConfig)


def _validate_guardrails_enum(value: str, allowed: Set[str], field_name: str) -> str:
    normalized = _as_str(value, "")
    if normalized not in allowed:
        raise ValueError(f"Invalid guardrails.{field_name} in config.json (allowed: {','.join(sorted(allowed))})")
    return normalized


def _parse_content_moderation_config(raw: Dict[str, Any], *, enabled_default: bool, field_prefix: str) -> OciContentModerationConfig:
    raw = raw or {}
    return OciContentModerationConfig(
        enabled=_as_bool(raw.get("enabled"), enabled_default),
        categories=_as_list_of_str(raw.get("categories"), ["OVERALL"]),
        threshold=_validate_threshold(raw.get("threshold", DEFAULT_SCORE_THRESHOLD), f"{field_prefix}.threshold"),
    )


def _parse_prompt_injection_config(raw: Dict[str, Any], *, enabled_default: bool, field_prefix: str) -> OciPromptInjectionConfig:
    raw = raw or {}
    return OciPromptInjectionConfig(
        enabled=_as_bool(raw.get("enabled"), enabled_default),
        threshold=_validate_threshold(raw.get("threshold", DEFAULT_SCORE_THRESHOLD), f"{field_prefix}.threshold"),
    )


def _parse_pii_detection_config(raw: Dict[str, Any], *, enabled_default: bool) -> OciPIIDetectionConfig:
    raw = raw or {}
    return OciPIIDetectionConfig(
        enabled=_as_bool(raw.get("enabled"), enabled_default),
        types=_as_list_of_str(raw.get("types"), []),
    )


def _parse_pii_rewrite_config(raw: Dict[str, Any]) -> GatewayPIIRewriteConfig:
    raw = raw or {}
    action = _as_str(raw.get("action"), "redact").lower()
    if action not in {"redact", "mask"}:
        raise ValueError("Invalid guardrails.gateway_extensions.output.pii_rewrite.action in config.json (allowed: redact,mask)")
    return GatewayPIIRewriteConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        action=action,
        placeholder=_as_str(raw.get("placeholder"), "[REDACTED]"),
    )


def _parse_local_blocklist_config(raw: Dict[str, Any], config_dir: Path) -> LocalBlocklistConfig:
    raw = raw or {}
    file_name = _as_str(raw.get("file"), "blocklist.txt")
    resolved = config_dir / file_name
    entries: Set[str] = set()
    enabled = _as_bool(raw.get("enabled"), False)
    if enabled and resolved.exists():
        entries = load_blocklist(resolved)
    elif enabled and not resolved.exists():
        logger.warning("Guardrails blocklist file not found: %s", resolved)
    return LocalBlocklistConfig(
        enabled=enabled,
        file=file_name,
        resolved_path=resolved,
        entries=entries,
    )


def build_guardrails_config(raw: Dict[str, Any], *, config_file_path: str) -> GuardrailsConfig:
    raw = raw or {}
    config_base_dir = Path(config_file_path).resolve().parent
    config_dir = config_base_dir / _as_str(raw.get("config_dir"), "guardrails")

    oci_native_raw = raw.get("oci_native", {}) or {}
    gateway_policy_raw = raw.get("gateway_policy", {}) or {}
    gateway_extensions_raw = raw.get("gateway_extensions", {}) or {}

    oci_input_raw = oci_native_raw.get("input", {}) or {}
    oci_output_raw = oci_native_raw.get("output", {}) or {}
    gateway_input_raw = gateway_extensions_raw.get("input", {}) or {}
    gateway_output_raw = gateway_extensions_raw.get("output", {}) or {}
    gateway_streaming_raw = gateway_extensions_raw.get("streaming", {}) or {}

    mode = _validate_guardrails_enum(
        _as_str(gateway_policy_raw.get("mode"), "block").lower(),
        {"block", "inform"},
        "gateway_policy.mode",
    )
    input_failure_mode = _validate_guardrails_enum(
        _as_str(gateway_policy_raw.get("input_failure_mode"), "closed").lower(),
        {"open", "closed"},
        "gateway_policy.input_failure_mode",
    )
    output_failure_mode = _validate_guardrails_enum(
        _as_str(gateway_policy_raw.get("output_failure_mode"), "open").lower(),
        {"open", "closed"},
        "gateway_policy.output_failure_mode",
    )
    streaming_behavior = _validate_guardrails_enum(
        _as_str(gateway_streaming_raw.get("when_output_guardrails_enabled"), "reject").lower(),
        {"reject", "downgrade_to_non_stream"},
        "gateway_extensions.streaming.when_output_guardrails_enabled",
    )

    block_http_status = int(gateway_policy_raw.get("block_http_status", 400))
    if block_http_status < 100 or block_http_status > 599:
        raise ValueError("Invalid guardrails.gateway_policy.block_http_status in config.json")

    oci_native = OciNativeGuardrailsConfig(
        default_language=_as_str(oci_native_raw.get("default_language"), "en"),
        input=OciNativeInputGuardrailsConfig(
            enabled=_as_bool(oci_input_raw.get("enabled"), True),
            content_moderation=_parse_content_moderation_config(
                oci_input_raw.get("content_moderation", {}),
                enabled_default=True,
                field_prefix="oci_native.input.content_moderation",
            ),
            prompt_injection=_parse_prompt_injection_config(
                oci_input_raw.get("prompt_injection", {}),
                enabled_default=True,
                field_prefix="oci_native.input.prompt_injection",
            ),
            pii_detection=_parse_pii_detection_config(
                oci_input_raw.get("pii_detection", {}),
                enabled_default=False,
            ),
        ),
        output=OciNativeOutputGuardrailsConfig(
            enabled=_as_bool(oci_output_raw.get("enabled"), False),
            content_moderation=_parse_content_moderation_config(
                oci_output_raw.get("content_moderation", {}),
                enabled_default=True,
                field_prefix="oci_native.output.content_moderation",
            ),
            pii_detection=_parse_pii_detection_config(
                oci_output_raw.get("pii_detection", {}),
                enabled_default=False,
            ),
        ),
    )

    gateway_policy = GatewayGuardrailsPolicyConfig(
        mode=mode,
        block_http_status=block_http_status,
        block_message=_as_str(gateway_policy_raw.get("block_message"), DEFAULT_BLOCK_MESSAGE),
        log_details=_as_bool(gateway_policy_raw.get("log_details"), False),
        redact_logs=_as_bool(gateway_policy_raw.get("redact_logs"), True),
        input_failure_mode=input_failure_mode,
        output_failure_mode=output_failure_mode,
    )

    gateway_extensions = GatewayGuardrailsExtensionsConfig(
        input=GatewayInputExtensionsConfig(
            include_system=_as_bool(gateway_input_raw.get("include_system"), False),
            include_tool_results=_as_bool(gateway_input_raw.get("include_tool_results"), True),
            local_blocklist=_parse_local_blocklist_config(gateway_input_raw.get("local_blocklist", {}), config_dir),
        ),
        output=GatewayOutputExtensionsConfig(
            pii_rewrite=_parse_pii_rewrite_config(gateway_output_raw.get("pii_rewrite", {})),
        ),
        streaming=GatewayStreamingExtensionsConfig(
            when_output_guardrails_enabled=streaming_behavior,
        ),
    )

    return GuardrailsConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        config_dir=config_dir,
        oci_native=oci_native,
        gateway_policy=gateway_policy,
        gateway_extensions=gateway_extensions,
    )


def load_blocklist(path: Path) -> Set[str]:
    entries: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            normalized = line.strip()
            if not normalized or normalized.startswith("#"):
                continue
            entries.add(normalized)
    return entries


def _extract_text_from_system(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: List[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return ""


def _extract_tool_result_text(block: Dict[str, Any]) -> str:
    result = block.get("content", block.get("result", ""))
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    return str(result or "")


def extract_text_from_message_content(content: Any, *, include_tool_results: bool) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "tool_result" and include_tool_results:
            tool_text = _extract_tool_result_text(block)
            if tool_text:
                parts.append(tool_text)
    return "\n".join(part for part in parts if part)


def collect_input_text_for_guardrails(body: Dict[str, Any], config: GuardrailsConfig) -> str:
    if not config.enabled or not config.oci_native.input.enabled:
        return ""

    parts: List[str] = []
    if config.gateway_extensions.input.include_system:
        system_text = _extract_text_from_system(body.get("system"))
        if system_text:
            parts.append(system_text)

    for message in body.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role != "user":
            continue
        text = extract_text_from_message_content(
            message.get("content"),
            include_tool_results=config.gateway_extensions.input.include_tool_results,
        )
        if text:
            parts.append(text)
    return "\n\n".join(part for part in parts if part)


def extract_text_blocks(content_blocks: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for block in content_blocks or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def replace_text_blocks(content_blocks: Sequence[Dict[str, Any]], new_text: str) -> List[Dict[str, Any]]:
    replaced: List[Dict[str, Any]] = []
    replaced_once = False
    for block in content_blocks or []:
        if isinstance(block, dict) and block.get("type") == "text":
            if not replaced_once:
                updated = dict(block)
                updated["text"] = new_text
                replaced.append(updated)
                replaced_once = True
            continue
        replaced.append(block)
    if not replaced_once:
        replaced.insert(0, {"type": "text", "text": new_text})
    return replaced


def check_local_blocklist(content: str, blocklist: Iterable[str]) -> List[str]:
    matches: List[str] = []
    haystack = content.lower()
    for entry in blocklist:
        if not entry:
            continue
        if entry.startswith("regex:"):
            pattern = entry[len("regex:"):].strip()
            if pattern and re.search(pattern, content, flags=re.IGNORECASE):
                matches.append(entry)
        else:
            if entry.lower() in haystack:
                matches.append(entry)
    return matches


def redact_pii_text(text: str, pii_entities: Sequence[Dict[str, Any]], *, action: str, placeholder: str) -> str:
    if not pii_entities:
        return text

    output = text
    for entity in sorted(pii_entities, key=lambda item: int(item.get("offset", 0)), reverse=True):
        offset = int(entity.get("offset", 0))
        length = int(entity.get("length", 0))
        entity_text = str(entity.get("text", ""))
        replacement = placeholder if action == "redact" else "*" * max(len(entity_text), length, 1)
        output = output[:offset] + replacement + output[offset + length:]
    return output


def _parse_category_scores(content_result: Any, threshold: float) -> ContentModerationFinding:
    categories = []
    triggered = False
    for item in getattr(content_result, "categories", []) or []:
        name = getattr(item, "name", "") or ""
        score = float(getattr(item, "score", 0.0) or 0.0)
        categories.append({"name": name, "score": score})
        if score >= threshold:
            triggered = True
    return ContentModerationFinding(categories=categories, triggered=triggered)


def _parse_prompt_injection(result_obj: Any, threshold: float) -> PromptInjectionFinding:
    score = float(getattr(result_obj, "score", 0.0) or 0.0)
    return PromptInjectionFinding(score=score, triggered=score >= threshold)


def _parse_pii(result_list: Sequence[Any]) -> PIIFinding:
    entities: List[Dict[str, Any]] = []
    for item in result_list or []:
        entities.append(
            {
                "text": str(getattr(item, "text", "") or ""),
                "label": str(getattr(item, "label", "") or ""),
                "score": float(getattr(item, "score", 0.0) or 0.0),
                "offset": int(getattr(item, "offset", 0) or 0),
                "length": int(getattr(item, "length", 0) or 0),
            }
        )
    return PIIFinding(entities=entities, detected=bool(entities))


def summarize_guardrails_result(result: GuardrailsCheckResult, *, redact_logs: bool) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "passed": result.passed,
        "issue_detected": result.issue_detected,
        "blocked_reason": result.blocked_reason,
        "local_blocklist_matches": list(result.local_blocklist_matches),
    }
    if result.content_moderation:
        summary["content_moderation"] = {
            "triggered": result.content_moderation.triggered,
            "categories": result.content_moderation.categories,
        }
    if result.prompt_injection:
        summary["prompt_injection"] = {
            "triggered": result.prompt_injection.triggered,
            "score": result.prompt_injection.score,
        }
    if result.pii:
        pii_entities = result.pii.entities
        if redact_logs:
            pii_entities = [
                {
                    "label": entity.get("label"),
                    "score": entity.get("score"),
                    "offset": entity.get("offset"),
                    "length": entity.get("length"),
                }
                for entity in pii_entities
            ]
        summary["pii"] = {
            "detected": result.pii.detected,
            "entities": pii_entities,
        }
    return summary


def _build_oci_guardrail_configs(
    *,
    content_moderation: Optional[OciContentModerationConfig],
    prompt_injection: Optional[OciPromptInjectionConfig],
    pii_detection: Optional[OciPIIDetectionConfig],
) -> Optional[oci.generative_ai_inference.models.GuardrailConfigs]:
    kwargs: Dict[str, Any] = {}
    if content_moderation and content_moderation.enabled:
        kwargs["content_moderation_config"] = oci.generative_ai_inference.models.ContentModerationConfiguration(
            categories=list(content_moderation.categories or ["OVERALL"])
        )
    if prompt_injection and prompt_injection.enabled:
        prompt_injection_cls = getattr(oci.generative_ai_inference.models, "PromptInjectionConfiguration", None)
        if prompt_injection_cls is None:
            raise GuardrailsSdkCompatibilityError(
                "Installed OCI SDK does not support PromptInjectionConfiguration. "
                "Upgrade the 'oci' package to a newer version that includes Guardrails prompt injection support."
            )
        kwargs["prompt_injection_config"] = prompt_injection_cls()
    if pii_detection and pii_detection.enabled and pii_detection.types:
        kwargs["personally_identifiable_information_config"] = (
            oci.generative_ai_inference.models.PersonallyIdentifiableInformationConfiguration(
                types=list(pii_detection.types)
            )
        )
    if not kwargs:
        return None
    return oci.generative_ai_inference.models.GuardrailConfigs(**kwargs)


async def _call_oci_apply_guardrails(
    *,
    client: Any,
    compartment_id: str,
    content: str,
    language_code: str,
    guardrail_configs: oci.generative_ai_inference.models.GuardrailConfigs,
) -> Any:
    details = oci.generative_ai_inference.models.ApplyGuardrailsDetails(
        input=oci.generative_ai_inference.models.GuardrailsTextInput(
            type="TEXT",
            content=content,
            language_code=language_code,
        ),
        guardrail_configs=guardrail_configs,
        compartment_id=compartment_id,
    )
    return await asyncio.to_thread(client.apply_guardrails, apply_guardrails_details=details)


async def apply_input_guardrails(
    *,
    client: Any,
    compartment_id: str,
    content: str,
    config: GuardrailsConfig,
    language_code: Optional[str] = None,
) -> GuardrailsCheckResult:
    result = GuardrailsCheckResult(passed=True)
    if not content or not config.enabled or not config.oci_native.input.enabled:
        return result

    if config.gateway_extensions.input.local_blocklist.enabled and config.gateway_extensions.input.local_blocklist.entries:
        matches = check_local_blocklist(content, config.gateway_extensions.input.local_blocklist.entries)
        if matches:
            result.issue_detected = True
            result.passed = False
            result.local_blocklist_matches = matches
            result.blocked_reason = "local_blocklist"

    oci_configs = _build_oci_guardrail_configs(
        content_moderation=config.oci_native.input.content_moderation,
        prompt_injection=config.oci_native.input.prompt_injection,
        pii_detection=config.oci_native.input.pii_detection,
    )
    if oci_configs is None:
        return result

    response = await _call_oci_apply_guardrails(
        client=client,
        compartment_id=compartment_id,
        content=content,
        language_code=language_code or config.oci_native.default_language,
        guardrail_configs=oci_configs,
    )
    guardrails_results = getattr(getattr(response, "data", None), "results", None)
    if guardrails_results is None:
        return result

    if config.oci_native.input.content_moderation.enabled:
        cm = _parse_category_scores(
            getattr(guardrails_results, "content_moderation", None),
            config.oci_native.input.content_moderation.threshold,
        )
        result.content_moderation = cm
        if cm.triggered:
            result.issue_detected = True
            result.passed = False
            result.blocked_reason = result.blocked_reason or "content_moderation"

    if config.oci_native.input.prompt_injection.enabled:
        pi = _parse_prompt_injection(
            getattr(guardrails_results, "prompt_injection", None),
            config.oci_native.input.prompt_injection.threshold,
        )
        result.prompt_injection = pi
        if pi.triggered:
            result.issue_detected = True
            result.passed = False
            result.blocked_reason = result.blocked_reason or "prompt_injection"

    if config.oci_native.input.pii_detection.enabled:
        pii = _parse_pii(getattr(guardrails_results, "personally_identifiable_information", None))
        result.pii = pii
        if pii.detected:
            result.issue_detected = True
            result.passed = False
            result.blocked_reason = result.blocked_reason or "pii"

    return result


async def apply_output_guardrails(
    *,
    client: Any,
    compartment_id: str,
    content: str,
    config: GuardrailsConfig,
    language_code: Optional[str] = None,
) -> GuardrailsCheckResult:
    result = GuardrailsCheckResult(passed=True)
    if not content or not config.enabled or not config.oci_native.output.enabled:
        return result

    oci_configs = _build_oci_guardrail_configs(
        content_moderation=config.oci_native.output.content_moderation,
        prompt_injection=None,
        pii_detection=config.oci_native.output.pii_detection,
    )
    if oci_configs is None:
        return result

    response = await _call_oci_apply_guardrails(
        client=client,
        compartment_id=compartment_id,
        content=content,
        language_code=language_code or config.oci_native.default_language,
        guardrail_configs=oci_configs,
    )
    guardrails_results = getattr(getattr(response, "data", None), "results", None)
    if guardrails_results is None:
        return result

    if config.oci_native.output.content_moderation.enabled:
        cm = _parse_category_scores(
            getattr(guardrails_results, "content_moderation", None),
            config.oci_native.output.content_moderation.threshold,
        )
        result.content_moderation = cm
        if cm.triggered:
            result.issue_detected = True
            result.passed = False
            result.blocked_reason = "content_moderation"

    if config.oci_native.output.pii_detection.enabled:
        pii = _parse_pii(getattr(guardrails_results, "personally_identifiable_information", None))
        result.pii = pii
        if pii.detected:
            result.issue_detected = True
            pii_rewrite = config.gateway_extensions.output.pii_rewrite
            if pii_rewrite.enabled:
                result.redacted_content = redact_pii_text(
                    content,
                    pii.entities,
                    action=pii_rewrite.action,
                    placeholder=pii_rewrite.placeholder,
                )

    return result
