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
DEFAULT_BLOCK_MESSAGE = "Request blocked by guardrails policy."


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


@dataclass
class ContentModerationConfig:
    enabled: bool = True
    categories: List[str] = field(default_factory=lambda: ["OVERALL"])
    threshold: float = DEFAULT_SCORE_THRESHOLD


@dataclass
class PromptInjectionConfig:
    enabled: bool = True
    threshold: float = DEFAULT_SCORE_THRESHOLD


@dataclass
class PIIConfig:
    enabled: bool = False
    types: List[str] = field(default_factory=list)
    action: str = "none"
    placeholder: str = "[REDACTED]"


@dataclass
class LocalBlocklistConfig:
    enabled: bool = False
    file: str = "blocklist.txt"
    resolved_path: Optional[Path] = None
    entries: Set[str] = field(default_factory=set)


@dataclass
class InputGuardrailsConfig:
    enabled: bool = True
    fail_mode: str = "closed"
    include_system: bool = False
    include_tool_results: bool = True
    content_moderation: ContentModerationConfig = field(default_factory=ContentModerationConfig)
    prompt_injection: PromptInjectionConfig = field(default_factory=PromptInjectionConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    local_blocklist: LocalBlocklistConfig = field(default_factory=LocalBlocklistConfig)


@dataclass
class OutputGuardrailsConfig:
    enabled: bool = False
    fail_mode: str = "open"
    content_moderation: ContentModerationConfig = field(default_factory=ContentModerationConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)


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
    mode: str = "block"
    default_language: str = "en"
    config_dir: Path = Path("guardrails")
    block_http_status: int = 400
    block_message: str = DEFAULT_BLOCK_MESSAGE
    streaming_behavior: str = "reject"
    log_details: bool = False
    redact_logs: bool = True
    input: InputGuardrailsConfig = field(default_factory=InputGuardrailsConfig)
    output: OutputGuardrailsConfig = field(default_factory=OutputGuardrailsConfig)


def _validate_guardrails_enum(value: str, allowed: Set[str], field_name: str) -> str:
    normalized = _as_str(value, "")
    if normalized not in allowed:
        raise ValueError(f"Invalid guardrails.{field_name} in config.json (allowed: {','.join(sorted(allowed))})")
    return normalized


def _parse_content_moderation_config(raw: Dict[str, Any], *, enabled_default: bool) -> ContentModerationConfig:
    raw = raw or {}
    return ContentModerationConfig(
        enabled=_as_bool(raw.get("enabled"), enabled_default),
        categories=_as_list_of_str(raw.get("categories"), ["OVERALL"]),
        threshold=float(raw.get("threshold", DEFAULT_SCORE_THRESHOLD)),
    )


def _parse_prompt_injection_config(raw: Dict[str, Any], *, enabled_default: bool) -> PromptInjectionConfig:
    raw = raw or {}
    return PromptInjectionConfig(
        enabled=_as_bool(raw.get("enabled"), enabled_default),
        threshold=float(raw.get("threshold", DEFAULT_SCORE_THRESHOLD)),
    )


def _parse_pii_config(raw: Dict[str, Any], *, enabled_default: bool, action_default: str) -> PIIConfig:
    raw = raw or {}
    action = _as_str(raw.get("action"), action_default).lower()
    if action not in {"none", "redact", "mask"}:
        raise ValueError("Invalid guardrails pii.action in config.json (allowed: none,redact,mask)")
    return PIIConfig(
        enabled=_as_bool(raw.get("enabled"), enabled_default),
        types=_as_list_of_str(raw.get("types"), []),
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

    mode = _validate_guardrails_enum(_as_str(raw.get("mode"), "block").lower(), {"block", "inform"}, "mode")
    streaming_behavior = _validate_guardrails_enum(
        _as_str(raw.get("streaming_behavior"), "reject").lower(),
        {"reject", "downgrade_to_non_stream"},
        "streaming_behavior",
    )

    block_http_status = int(raw.get("block_http_status", 400))
    if block_http_status < 100 or block_http_status > 599:
        raise ValueError("Invalid guardrails.block_http_status in config.json")

    input_raw = raw.get("input", {}) or {}
    output_raw = raw.get("output", {}) or {}

    input_fail_mode = _validate_guardrails_enum(
        _as_str(input_raw.get("fail_mode"), "closed").lower(),
        {"open", "closed"},
        "input.fail_mode",
    )
    output_fail_mode = _validate_guardrails_enum(
        _as_str(output_raw.get("fail_mode"), "open").lower(),
        {"open", "closed"},
        "output.fail_mode",
    )

    input_config = InputGuardrailsConfig(
        enabled=_as_bool(input_raw.get("enabled"), True),
        fail_mode=input_fail_mode,
        include_system=_as_bool(input_raw.get("include_system"), False),
        include_tool_results=_as_bool(input_raw.get("include_tool_results"), True),
        content_moderation=_parse_content_moderation_config(
            input_raw.get("content_moderation", {}), enabled_default=True
        ),
        prompt_injection=_parse_prompt_injection_config(
            input_raw.get("prompt_injection", {}), enabled_default=True
        ),
        pii=_parse_pii_config(input_raw.get("pii", {}), enabled_default=False, action_default="none"),
        local_blocklist=_parse_local_blocklist_config(input_raw.get("local_blocklist", {}), config_dir),
    )

    output_config = OutputGuardrailsConfig(
        enabled=_as_bool(output_raw.get("enabled"), False),
        fail_mode=output_fail_mode,
        content_moderation=_parse_content_moderation_config(
            output_raw.get("content_moderation", {}), enabled_default=True
        ),
        pii=_parse_pii_config(output_raw.get("pii", {}), enabled_default=False, action_default="redact"),
    )

    return GuardrailsConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        mode=mode,
        default_language=_as_str(raw.get("default_language"), "en"),
        config_dir=config_dir,
        block_http_status=block_http_status,
        block_message=_as_str(raw.get("block_message"), DEFAULT_BLOCK_MESSAGE),
        streaming_behavior=streaming_behavior,
        log_details=_as_bool(raw.get("log_details"), False),
        redact_logs=_as_bool(raw.get("redact_logs"), True),
        input=input_config,
        output=output_config,
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
    if not config.enabled or not config.input.enabled:
        return ""

    parts: List[str] = []
    if config.input.include_system:
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
            include_tool_results=config.input.include_tool_results,
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
    if action == "none" or not pii_entities:
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
    content_moderation: Optional[ContentModerationConfig],
    prompt_injection: Optional[PromptInjectionConfig],
    pii: Optional[PIIConfig],
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
    if pii and pii.enabled and pii.types:
        kwargs["personally_identifiable_information_config"] = (
            oci.generative_ai_inference.models.PersonallyIdentifiableInformationConfiguration(
                types=list(pii.types)
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
    if not content or not config.enabled or not config.input.enabled:
        return result

    if config.input.local_blocklist.enabled and config.input.local_blocklist.entries:
        matches = check_local_blocklist(content, config.input.local_blocklist.entries)
        if matches:
            result.issue_detected = True
            result.passed = False
            result.local_blocklist_matches = matches
            result.blocked_reason = "local_blocklist"

    oci_configs = _build_oci_guardrail_configs(
        content_moderation=config.input.content_moderation,
        prompt_injection=config.input.prompt_injection,
        pii=config.input.pii,
    )
    if oci_configs is None:
        return result

    response = await _call_oci_apply_guardrails(
        client=client,
        compartment_id=compartment_id,
        content=content,
        language_code=language_code or config.default_language,
        guardrail_configs=oci_configs,
    )
    guardrails_results = getattr(getattr(response, "data", None), "results", None)
    if guardrails_results is None:
        return result

    if config.input.content_moderation.enabled:
        cm = _parse_category_scores(
            getattr(guardrails_results, "content_moderation", None),
            config.input.content_moderation.threshold,
        )
        result.content_moderation = cm
        if cm.triggered:
            result.issue_detected = True
            result.passed = False
            result.blocked_reason = result.blocked_reason or "content_moderation"

    if config.input.prompt_injection.enabled:
        pi = _parse_prompt_injection(
            getattr(guardrails_results, "prompt_injection", None),
            config.input.prompt_injection.threshold,
        )
        result.prompt_injection = pi
        if pi.triggered:
            result.issue_detected = True
            result.passed = False
            result.blocked_reason = result.blocked_reason or "prompt_injection"

    if config.input.pii.enabled:
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
    if not content or not config.enabled or not config.output.enabled:
        return result

    oci_configs = _build_oci_guardrail_configs(
        content_moderation=config.output.content_moderation,
        prompt_injection=None,
        pii=config.output.pii,
    )
    if oci_configs is None:
        return result

    response = await _call_oci_apply_guardrails(
        client=client,
        compartment_id=compartment_id,
        content=content,
        language_code=language_code or config.default_language,
        guardrail_configs=oci_configs,
    )
    guardrails_results = getattr(getattr(response, "data", None), "results", None)
    if guardrails_results is None:
        return result

    if config.output.content_moderation.enabled:
        cm = _parse_category_scores(
            getattr(guardrails_results, "content_moderation", None),
            config.output.content_moderation.threshold,
        )
        result.content_moderation = cm
        if cm.triggered:
            result.issue_detected = True
            result.passed = False
            result.blocked_reason = "content_moderation"

    if config.output.pii.enabled:
        pii = _parse_pii(getattr(guardrails_results, "personally_identifiable_information", None))
        result.pii = pii
        if pii.detected:
            result.issue_detected = True
            if config.output.pii.action in {"redact", "mask"}:
                result.redacted_content = redact_pii_text(
                    content,
                    pii.entities,
                    action=config.output.pii.action,
                    placeholder=config.output.pii.placeholder,
                )

    return result
