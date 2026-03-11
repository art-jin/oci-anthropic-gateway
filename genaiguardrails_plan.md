# OCI-Anthropic Gateway Guardrails 集成计划

## 文档版本
- **日期**: 2026-03-11
- **状态**: 待审批
- **目标**: 集成 OCI Guardrails API，支持 Input/Output 内容审核

---

## 一、文件变更概览

### 新建文件（6 个）

| 文件路径 | 用途 |
|----------|------|
| `src/utils/guardrails.py` | Guardrails 核心逻辑模块 |
| `guardrails/blocklist.txt` | 过滤词列表 |
| `guardrails/blocklist.txt.template` | 过滤词列表模板 |
| `guardrails/pii_config.json` | PII 检测配置 |
| `guardrails/pii_config.json.template` | PII 配置模板 |
| `tests/test_guardrails.py` | 单元测试 |

### 修改文件（4 个）

| 文件路径 | 修改类型 |
|----------|----------|
| `config.json.template` | 新增 guardrails 配置节 |
| `src/config/__init__.py` | 解析 guardrails 配置，新增 GuardrailsConfig 类 |
| `src/routes/handlers.py` | 集成 Guardrails 检查到请求流程 |
| `.gitignore` | 忽略 `guardrails/blocklist.txt` 和 `guardrails/pii_config.json` |

---

## 二、详细修改计划

### 2.1 新建 `config.json.template` 修改

**位置**: 根目录
**修改内容**: 在现有配置末尾、`server` 节之前添加 `guardrails` 配置节

```json
{
  ...现有配置...,

  "guardrails": {
    "enabled": false,
    "mode": "block",
    "config_dir": "guardrails",
    "input": {
      "content_moderation": {
        "enabled": true,
        "categories": ["OVERALL", "BLOCKLIST"],
        "blocklist_file": "blocklist.txt"
      },
      "prompt_injection": {
        "enabled": true
      },
      "pii": {
        "enabled": false,
        "config_file": "pii_config.json"
      }
    },
    "output": {
      "enabled": false,
      "content_moderation": {
        "enabled": true,
        "categories": ["OVERALL"]
      },
      "pii": {
        "enabled": false,
        "config_file": "pii_config.json",
        "action": "redact"
      }
    },
    "default_language": "en",
    "block_http_status": 400,
    "block_message": "Inappropriate content detected. Please revise your request.",
    "inform_log_level": "warning"
  },

  "server": {...}
}
```

**配置字段说明**:

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | Guardrails 总开关 |
| `mode` | string | "block" | "block" 拦截请求 / "inform" 仅记录日志 |
| `config_dir` | string | "guardrails" | Guardrails 配置文件目录 |
| `input` | object | - | 输入检查配置 |
| `output` | object | - | 输出检查配置 |
| `default_language` | string | "en" | 默认语言代码 |
| `block_http_status` | int | 400 | 拦截时返回的 HTTP 状态码 |
| `block_message` | string | - | 拦截时返回的错误消息 |
| `inform_log_level` | string | "warning" | inform 模式下的日志级别 |

---

### 2.2 新建 `guardrails/blocklist.txt.template`

**位置**: `guardrails/blocklist.txt.template`
**内容**:

```
# OCI Guardrails Blocklist Configuration
#
# Format: One word/phrase per line
# Lines starting with # are comments and will be ignored
# Empty lines will be ignored
#
# Examples:
# sensitive_word
# competitor_name
#
# Regex support (line starts with "regex:"):
# regex:\d{4}-\d{4}-\d{4}-\d{4}
# regex:[A-Z]{2}\d{6}
#
# Note: When using BLOCKLIST category in OCI ApplyGuardrails API,
#       OCI uses its predefined blocklist. This file is for
#       additional application-level filtering if needed.

# Add your custom blocklist entries below:
```

---

### 2.3 新建 `guardrails/pii_config.json.template`

**位置**: `guardrails/pii_config.json.template`
**内容**:

```json
{
  "enabled_types": [
    "PERSON",
    "EMAIL",
    "TELEPHONE_NUMBER",
    "CREDIT_CARD_NUMBER",
    "IP_ADDRESS",
    "ADDRESS",
    "DATE_TIME",
    "LOCATION"
  ],
  "action": "redact",
  "redaction_placeholder": "[REDACTED]",
  "confidence_threshold": 0.85,
  "per_type_overrides": {
    "EMAIL": {
      "action": "mask",
      "mask_char": "*",
      "preserve_domain": true
    },
    "TELEPHONE_NUMBER": {
      "action": "redact"
    }
  }
}
```

**配置字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled_types` | array | 要检测的 PII 类型列表 |
| `action` | string | 默认处理方式: "redact" / "mask" / "none" |
| `redaction_placeholder` | string | 遮蔽后的占位符 |
| `confidence_threshold` | float | 置信度阈值 (0.0-1.0) |
| `per_type_overrides` | object | 按类型覆盖配置 |

---

### 2.4 新建 `src/utils/guardrails.py`

**位置**: `src/utils/guardrails.py`
**职责**: Guardrails 核心逻辑

**模块结构设计**:

```python
"""
Guardrails integration for OCI Anthropic Gateway.

Provides content moderation, prompt injection detection, and PII handling
using OCI Generative AI ApplyGuardrails API.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import oci.generative_ai_inference.models

logger = logging.getLogger("oci-gateway")


# === Enums ===

class GuardrailMode(Enum):
    BLOCK = "block"
    INFORM = "inform"


class PIIAction(Enum):
    REDACT = "redact"
    MASK = "mask"
    NONE = "none"


# === Data Classes ===

@dataclass
class ContentModerationResult:
    """Content moderation check result."""
    overall_score: float = 0.0
    blocklist_score: float = 0.0
    is_unsafe: bool = False
    categories: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PromptInjectionResult:
    """Prompt injection check result."""
    score: float = 0.0
    is_injection: bool = False


@dataclass
class PIIEntity:
    """Single PII detection entity."""
    text: str
    label: str
    score: float
    offset: int
    length: int


@dataclass
class PIIResult:
    """PII detection result."""
    entities: List[PIIEntity] = field(default_factory=list)
    has_pii: bool = False


@dataclass
class GuardrailsCheckResult:
    """Combined result of all guardrails checks."""
    passed: bool = True
    content_moderation: Optional[ContentModerationResult] = None
    prompt_injection: Optional[PromptInjectionResult] = None
    pii: Optional[PIIResult] = None
    blocked_reason: Optional[str] = None
    redacted_content: Optional[str] = None


# === Configuration Classes ===

@dataclass
class PIITypesConfig:
    """PII detection configuration."""
    enabled_types: List[str] = field(default_factory=list)
    action: str = "redact"
    redaction_placeholder: str = "[REDACTED]"
    confidence_threshold: float = 0.85
    per_type_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ContentModerationConfig:
    """Content moderation configuration."""
    enabled: bool = True
    categories: List[str] = field(default_factory=lambda: ["OVERALL"])
    blocklist_file: Optional[str] = None


@dataclass
class PromptInjectionConfig:
    """Prompt injection detection configuration."""
    enabled: bool = True


@dataclass
class InputGuardrailsConfig:
    """Input guardrails configuration."""
    content_moderation: ContentModerationConfig = field(default_factory=ContentModerationConfig)
    prompt_injection: PromptInjectionConfig = field(default_factory=PromptInjectionConfig)
    pii: Optional[PIITypesConfig] = None


@dataclass
class OutputGuardrailsConfig:
    """Output guardrails configuration."""
    enabled: bool = False
    content_moderation: ContentModerationConfig = field(default_factory=ContentModerationConfig)
    pii: Optional[PIITypesConfig] = None


@dataclass
class GuardrailsConfig:
    """Main guardrails configuration."""
    enabled: bool = False
    mode: str = "block"
    config_dir: str = "guardrails"
    input: InputGuardrailsConfig = field(default_factory=InputGuardrailsConfig)
    output: OutputGuardrailsConfig = field(default_factory=OutputGuardrailsConfig)
    default_language: str = "en"
    block_http_status: int = 400
    block_message: str = "Inappropriate content detected."
    inform_log_level: str = "warning"

    # Runtime loaded data
    blocklist: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Load external config files after initialization."""
        self._load_blocklist()
        self._load_pii_config()

    def _load_blocklist(self) -> None:
        """Load blocklist from file."""
        pass  # Implementation details

    def _load_pii_config(self) -> None:
        """Load PII config from file."""
        pass  # Implementation details


# === Core Functions ===

def extract_text_from_messages(messages: List[Any]) -> str:
    """Extract plain text from messages for guardrails checking.

    Args:
        messages: List of OCI or Anthropic format messages

    Returns:
        Concatenated text content
    """
    pass  # Implementation details


def apply_input_guardrails(
    client: Any,
    content: str,
    config: GuardrailsConfig,
    language_code: Optional[str] = None
) -> GuardrailsCheckResult:
    """Apply guardrails to input content.

    Args:
        client: OCI GenerativeAiInferenceClient
        content: Text content to check
        config: Guardrails configuration
        language_code: Language code (defaults to config.default_language)

    Returns:
        GuardrailsCheckResult with check results
    """
    pass  # Implementation details


def apply_output_guardrails(
    client: Any,
    content: str,
    config: GuardrailsConfig,
    language_code: Optional[str] = None
) -> GuardrailsCheckResult:
    """Apply guardrails to output content.

    Args:
        client: OCI GenerativeAiInferenceClient
        content: Text content to check
        config: Guardrails configuration
        language_code: Language code

    Returns:
        GuardrailsCheckResult with check results and optionally redacted content
    """
    pass  # Implementation details


def redact_pii(content: str, pii_result: PIIResult, config: PIITypesConfig) -> str:
    """Redact PII entities from content.

    Args:
        content: Original text content
        pii_result: PII detection result
        config: PII configuration

    Returns:
        Content with PII redacted/masked
    """
    pass  # Implementation details


def check_local_blocklist(content: str, blocklist: Set[str]) -> List[str]:
    """Check content against local blocklist.

    Args:
        content: Text to check
        blocklist: Set of blocked words/phrases

    Returns:
        List of matched blocklist entries
    """
    pass  # Implementation details


def build_guardrail_configs_for_oci(
    config: GuardrailsConfig,
    check_type: str  # "input" or "output"
) -> oci.generative_ai_inference.models.GuardrailConfigs:
    """Build OCI GuardrailConfigs object from our config.

    Args:
        config: Our guardrails configuration
        check_type: "input" or "output"

    Returns:
        OCI GuardrailConfigs object
    """
    pass  # Implementation details
```

---

### 2.5 修改 `src/config/__init__.py`

**位置**: `src/config/__init__.py`
**修改内容**:

#### 2.5.1 新增导入（文件顶部）

```python
from ..utils.guardrails import (
    GuardrailsConfig,
    InputGuardrailsConfig,
    OutputGuardrailsConfig,
    ContentModerationConfig,
    PromptInjectionConfig,
    PIITypesConfig,
)
```

#### 2.5.2 新增 Config 类属性（`__init__` 方法中）

在现有属性后添加:

```python
# Guardrails configuration
self.guardrails_enabled: bool = False
self.guardrails_mode: str = "block"
self.guardrails_config_dir: str = "guardrails"
self.guardrails_input_config: Optional[InputGuardrailsConfig] = None
self.guardrails_output_config: Optional[OutputGuardrailsConfig] = None
self.guardrails_default_language: str = "en"
self.guardrails_block_http_status: int = 400
self.guardrails_block_message: str = "Inappropriate content detected."
self.guardrails_inform_log_level: str = "warning"
self.guardrails: Optional[GuardrailsConfig] = None
```

#### 2.5.3 新增配置加载逻辑（`_load_config` 方法中）

在 `server` 配置解析之后添加:

```python
# Load guardrails configuration
guardrails_conf = custom_config.get("guardrails", {})
self.guardrails_enabled = bool(guardrails_conf.get("enabled", False))

if self.guardrails_enabled:
    self.guardrails_mode = str(guardrails_conf.get("mode", "block"))
    self.guardrails_config_dir = str(guardrails_conf.get("config_dir", "guardrails"))
    self.guardrails_default_language = str(guardrails_conf.get("default_language", "en"))
    self.guardrails_block_http_status = int(guardrails_conf.get("block_http_status", 400))
    self.guardrails_block_message = str(guardrails_conf.get("block_message", "Inappropriate content detected."))
    self.guardrails_inform_log_level = str(guardrails_conf.get("inform_log_level", "warning"))

    # Parse input config
    input_conf = guardrails_conf.get("input", {})
    cm_input_conf = input_conf.get("content_moderation", {})
    pi_input_conf = input_conf.get("prompt_injection", {})
    pii_input_conf = input_conf.get("pii", {})

    input_content_moderation = ContentModerationConfig(
        enabled=bool(cm_input_conf.get("enabled", True)),
        categories=cm_input_conf.get("categories", ["OVERALL"]),
        blocklist_file=cm_input_conf.get("blocklist_file")
    )

    input_prompt_injection = PromptInjectionConfig(
        enabled=bool(pi_input_conf.get("enabled", True))
    )

    input_pii = None
    if pii_input_conf.get("enabled", False):
        input_pii = PIITypesConfig(
            config_file=pii_input_conf.get("config_file"),
            # Other fields loaded from file
        )

    self.guardrails_input_config = InputGuardrailsConfig(
        content_moderation=input_content_moderation,
        prompt_injection=input_prompt_injection,
        pii=input_pii
    )

    # Parse output config
    output_conf = guardrails_conf.get("output", {})
    self.guardrails_output_enabled = bool(output_conf.get("enabled", False))

    # ... similar parsing for output config

    # Build final GuardrailsConfig
    self.guardrails = GuardrailsConfig(
        enabled=self.guardrails_enabled,
        mode=self.guardrails_mode,
        config_dir=self.guardrails_config_dir,
        input=self.guardrails_input_config,
        output=self.guardrails_output_config,
        default_language=self.guardrails_default_language,
        block_http_status=self.guardrails_block_http_status,
        block_message=self.guardrails_block_message,
        inform_log_level=self.guardrails_inform_log_level
    )

    logger.info(
        f"Guardrails enabled: mode={self.guardrails_mode} "
        f"input_cm={input_content_moderation.enabled} "
        f"input_pi={input_prompt_injection.enabled} "
        f"output={self.guardrails_output_enabled}"
    )
```

#### 2.5.4 新增 Guardrails 目录验证

```python
# Validate guardrails config directory exists
if self.guardrails_enabled:
    guardrails_path = Path(self.guardrails_config_dir)
    if not guardrails_path.exists():
        logger.warning(f"Guardrails config directory not found: {self.guardrails_config_dir}")
        # Optionally create it
        guardrails_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created guardrails config directory: {self.guardrails_config_dir}")
```

---

### 2.6 修改 `src/routes/handlers.py`

**位置**: `src/routes/handlers.py`
**修改内容**:

#### 2.6.1 新增导入（文件顶部）

```python
from ..utils.guardrails import (
    apply_input_guardrails,
    apply_output_guardrails,
    extract_text_from_messages,
    GuardrailsCheckResult,
)
```

#### 2.6.2 修改 `handle_messages_request` 函数

在消息验证之后、OCI 调用之前添加 Input Guardrails 检查:

```python
async def handle_messages_request(
    body: dict,
    req_model: str,
    app_config
) -> Union[StreamingResponse, JSONResponse]:
    """Handle messages API endpoint."""

    # ... 现有验证逻辑 ...

    # === 新增: Input Guardrails 检查 ===
    if app_config.guardrails_enabled and app_config.guardrails:
        # 提取用户消息文本
        user_messages = [m for m in messages if m.get("role") == "user"]
        input_text = extract_text_from_messages(user_messages)

        if input_text:
            input_result = await apply_input_guardrails(
                client=app_config.genai_client,
                content=input_text,
                config=app_config.guardrails,
                language_code=None  # 使用默认语言
            )

            if not input_result.passed:
                logger.warning(
                    f"Input guardrails blocked request: {input_result.blocked_reason}"
                )

                if app_config.guardrails_mode == "block":
                    return JSONResponse(
                        status_code=app_config.guardrails_block_http_status,
                        content={
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": app_config.guardrails_block_message
                            }
                        }
                    )
                else:
                    # inform 模式: 记录日志但继续处理
                    log_func = getattr(logger, app_config.guardrails_inform_log_level, logger.warning)
                    log_func(f"Guardrails detected issues: {input_result.blocked_reason}")

    # ... 现有消息转换和 OCI 调用逻辑 ...

    # === 新增: Output Guardrails 检查（仅非流式） ===
    if not body.get("stream", False):
        if app_config.guardrails_enabled and app_config.guardrails:
            if app_config.guardrails.output and app_config.guardrails.output.enabled:
                # 提取响应文本
                output_text = extract_output_text(response)  # 需要实现

                if output_text:
                    output_result = await apply_output_guardrails(
                        client=app_config.genai_client,
                        content=output_text,
                        config=app_config.guardrails
                    )

                    if not output_result.passed:
                        if output_result.redacted_content:
                            # 替换响应中的内容
                            response = update_response_content(response, output_result.redacted_content)

    # ... 返回响应 ...
```

#### 2.6.3 处理流式响应的 Output Guardrails

对于流式响应，需要缓冲完整输出后再检查:

```python
async def generate_oci_stream_with_guardrails(
    oci_msgs,
    params_with_tools,
    message_id,
    model_conf,
    req_model,
    genai_client,
    cohere_messages,
    debug_enabled,
    debug_redact_media,
    trace_ctx,
    guardrails_config  # 新增参数
):
    """Streaming generation with output guardrails support."""

    if not guardrails_config or not guardrails_config.output or not guardrails_config.output.enabled:
        # Guardrails 未启用，使用原有逻辑
        async for chunk in generate_oci_stream(...):
            yield chunk
        return

    # 缓冲完整响应
    full_content = []
    async for chunk in generate_oci_stream(...):
        # 提取文本内容
        text = extract_text_from_sse_chunk(chunk)
        if text:
            full_content.append(text)
        yield chunk

    # 在流结束后检查（可选：通过 SSE 事件发送警告）
    complete_text = "".join(full_content)
    if complete_text:
        result = await apply_output_guardrails(...)
        if not result.passed:
            # 发送 guardrails 警告事件
            yield format_sse_event("guardrails_warning", {
                "message": "Output content flagged by guardrails",
                "details": result.blocked_reason
            })
```

---

### 2.7 修改 `.gitignore`

**位置**: `.gitignore`
**新增内容**:

```gitignore
# Guardrails configuration (may contain sensitive blocklists)
guardrails/blocklist.txt
guardrails/pii_config.json
```

---

### 2.8 新建 `tests/test_guardrails.py`

**位置**: `tests/test_guardrails.py`
**内容结构**:

```python
"""
Unit tests for Guardrails integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.utils.guardrails import (
    GuardrailsConfig,
    GuardrailsCheckResult,
    ContentModerationResult,
    PromptInjectionResult,
    PIIResult,
    PIIEntity,
    apply_input_guardrails,
    apply_output_guardrails,
    extract_text_from_messages,
    redact_pii,
    check_local_blocklist,
)


class TestExtractTextFromMessages:
    """Tests for extract_text_from_messages function."""

    def test_extract_simple_text(self):
        """Test extracting text from simple text messages."""
        pass

    def test_extract_multimodal_messages(self):
        """Test extracting text from messages with images."""
        pass

    def test_extract_empty_messages(self):
        """Test handling empty message list."""
        pass


class TestCheckLocalBlocklist:
    """Tests for local blocklist checking."""

    def test_no_match(self):
        """Test content with no blocklist matches."""
        pass

    def test_single_match(self):
        """Test content with single blocklist match."""
        pass

    def test_multiple_matches(self):
        """Test content with multiple blocklist matches."""
        pass

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        pass

    def test_regex_pattern(self):
        """Test regex pattern matching."""
        pass


class TestRedactPII:
    """Tests for PII redaction."""

    def test_redact_email(self):
        """Test email redaction."""
        pass

    def test_redact_phone(self):
        """Test phone number redaction."""
        pass

    def test_mask_email_preserve_domain(self):
        """Test email masking with domain preservation."""
        pass

    def test_multiple_pii_types(self):
        """Test redacting multiple PII types."""
        pass


class TestApplyInputGuardrails:
    """Tests for input guardrails."""

    @pytest.mark.asyncio
    async def test_clean_content_passes(self):
        """Test that clean content passes all checks."""
        pass

    @pytest.mark.asyncio
    async def test_content_moderation_block(self):
        """Test that inappropriate content is blocked."""
        pass

    @pytest.mark.asyncio
    async def test_prompt_injection_detection(self):
        """Test prompt injection detection."""
        pass

    @pytest.mark.asyncio
    async def test_pii_detection(self):
        """Test PII detection in input."""
        pass

    @pytest.mark.asyncio
    async def test_inform_mode(self):
        """Test that inform mode doesn't block."""
        pass


class TestApplyOutputGuardrails:
    """Tests for output guardrails."""

    @pytest.mark.asyncio
    async def test_clean_output_passes(self):
        """Test that clean output passes."""
        pass

    @pytest.mark.asyncio
    async def test_output_pii_redaction(self):
        """Test PII redaction in output."""
        pass

    @pytest.mark.asyncio
    async def test_output_content_moderation(self):
        """Test content moderation on output."""
        pass


class TestGuardrailsConfig:
    """Tests for GuardrailsConfig class."""

    def test_load_blocklist_from_file(self):
        """Test loading blocklist from file."""
        pass

    def test_load_pii_config_from_file(self):
        """Test loading PII config from file."""
        pass

    def test_missing_config_file_handling(self):
        """Test handling of missing config files."""
        pass

    def test_disabled_guardrails(self):
        """Test behavior when guardrails disabled."""
        pass


class TestIntegration:
    """Integration tests with mocked OCI client."""

    @pytest.mark.asyncio
    async def test_full_input_flow(self):
        """Test complete input guardrails flow."""
        pass

    @pytest.mark.asyncio
    async def test_full_output_flow(self):
        """Test complete output guardrails flow."""
        pass
```

---

## 三、实现顺序

```
Phase 1: 基础设施 (Day 1 上午)
├── 1.1 创建 guardrails/ 目录
├── 1.2 创建 blocklist.txt.template
├── 1.3 创建 pii_config.json.template
├── 1.4 更新 .gitignore
└── 1.5 更新 config.json.template

Phase 2: 核心模块 (Day 1 下午 - Day 2)
├── 2.1 创建 src/utils/guardrails.py
│   ├── 2.1.1 数据类定义
│   ├── 2.1.2 配置加载逻辑
│   ├── 2.1.3 文本提取函数
│   ├── 2.1.4 OCI API 调用封装
│   ├── 2.1.5 PII 遮蔽逻辑
│   └── 2.1.6 本地过滤词检查
└── 2.2 修改 src/config/__init__.py
    ├── 2.2.1 新增属性
    ├── 2.2.2 解析逻辑
    └── 2.2.3 配置验证

Phase 3: 集成到请求流程 (Day 3)
├── 3.1 修改 src/routes/handlers.py
│   ├── 3.1.1 Input Guardrails 集成
│   ├── 3.1.2 Output Guardrails 集成 (非流式)
│   └── 3.1.3 错误响应格式化
└── 3.2 修改 src/services/generation.py (流式支持)
    ├── 3.2.1 流式缓冲逻辑
    └── 3.2.2 流式 guardrails 事件

Phase 4: 测试 (Day 3 下午 - Day 4)
├── 4.1 单元测试
│   ├── 4.1.1 文本提取测试
│   ├── 4.1.2 过滤词检查测试
│   ├── 4.1.3 PII 遮蔽测试
│   └── 4.1.4 配置加载测试
├── 4.2 集成测试
│   ├── 4.2.1 端到端 input flow
│   └── 4.2.2 端到端 output flow
└── 4.3 手动测试
    ├── 4.3.1 block 模式测试
    ├── 4.3.2 inform 模式测试
    └── 4.3.3 流式响应测试

Phase 5: 文档更新 (Day 4)
├── 5.1 更新 CLAUDE.md
├── 5.2 更新 README.md
└── 5.3 创建 GUARDRAILS.md (详细使用文档)
```

---

## 四、API 调用流程图

### 4.1 Input Guardrails 流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         handle_messages_request()                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   guardrails.enabled == true? │
                    └───────────────────────────────┘
                           │              │
                          Yes             No
                           │              │
                           ▼              │
          ┌────────────────────────────────┐
          │ extract_text_from_messages()   │
          │ (提取 user 消息文本)            │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │ apply_input_guardrails()       │
          │                                │
          │  1. 本地过滤词检查 (可选)        │
          │  2. 调用 OCI ApplyGuardrails   │
          │  3. 解析结果                    │
          └────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ passed?     │
                    └─────────────┘
                      │         │
                     Yes        No
                      │         │
                      │         ▼
                      │    ┌─────────────────┐
                      │    │ mode == "block"? │
                      │    └─────────────────┘
                      │        │         │
                      │       Yes        No (inform)
                      │        │         │
                      │        ▼         ▼
                      │   ┌─────────┐  ┌──────────┐
                      │   │ 返回    │  │ 记录日志  │
                      │   │ 400错误 │  │ 继续处理  │
                      │   └─────────┘  └──────────┘
                      │
                      ▼
          ┌────────────────────────────────┐
          │ 继续正常请求处理流程             │
          │ (消息转换、OCI 推理等)           │
          └────────────────────────────────┘
```

### 4.2 Output Guardrails 流程 (非流式)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OCI 推理完成，获得响应                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │ guardrails.output.enabled == true?│
                    └───────────────────────────────────┘
                           │              │
                          Yes             No
                           │              │
                           ▼              │
          ┌────────────────────────────────┐
          │ extract_output_text(response)  │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │ apply_output_guardrails()      │
          │                                │
          │  1. Content Moderation 检查    │
          │  2. PII 检测与遮蔽             │
          └────────────────────────────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │ passed?         │
                    └─────────────────┘
                      │         │
                     Yes        No
                      │         │
                      │         ▼
                      │    ┌─────────────────────────┐
                      │    │ redacted_content 存在?  │
                      │    └─────────────────────────┘
                      │        │         │
                      │       Yes        No
                      │        │         │
                      │        ▼         ▼
                      │   ┌──────────┐ ┌────────────┐
                      │   │ 替换响应  │ │ 返回错误/  │
                      │   │ 内容      │ │ 默认消息    │
                      │   └──────────┘ └────────────┘
                      │
                      ▼
          ┌────────────────────────────────┐
          │ 返回处理后的响应给客户端         │
          └────────────────────────────────┘
```

---

## 五、错误响应格式

### 5.1 Block 模式错误响应

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Inappropriate content detected. Please revise your request."
  }
}
```

### 5.2 详细错误响应 (可选，调试用)

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Inappropriate content detected.",
    "details": {
      "content_moderation": {
        "overall_score": 1.0,
        "categories": ["HATE", "HARASSMENT"]
      },
      "prompt_injection": {
        "score": 0.0
      },
      "pii": {
        "detected_types": ["EMAIL"],
        "count": 1
      }
    }
  }
}
```

### 5.3 流式响应中的 Guardrails 警告事件

```
event: guardrails_warning
data: {"type": "guardrails_warning", "message": "Output flagged", "action": "logged"}
```

---

## 六、配置示例

### 6.1 最小配置 (仅启用 Prompt Injection 防护)

```json
{
  "guardrails": {
    "enabled": true,
    "mode": "block",
    "input": {
      "content_moderation": { "enabled": false },
      "prompt_injection": { "enabled": true },
      "pii": { "enabled": false }
    },
    "output": { "enabled": false }
  }
}
```

### 6.2 完整配置 (全部启用)

```json
{
  "guardrails": {
    "enabled": true,
    "mode": "block",
    "config_dir": "guardrails",
    "input": {
      "content_moderation": {
        "enabled": true,
        "categories": ["OVERALL", "BLOCKLIST"],
        "blocklist_file": "blocklist.txt"
      },
      "prompt_injection": { "enabled": true },
      "pii": {
        "enabled": true,
        "config_file": "pii_config.json"
      }
    },
    "output": {
      "enabled": true,
      "content_moderation": {
        "enabled": true,
        "categories": ["OVERALL"]
      },
      "pii": {
        "enabled": true,
        "config_file": "pii_config.json",
        "action": "redact"
      }
    },
    "default_language": "en",
    "block_http_status": 400,
    "block_message": "Your request could not be processed due to content policy violations.",
    "inform_log_level": "warning"
  }
}
```

### 6.3 Inform 模式 (仅记录，不拦截)

```json
{
  "guardrails": {
    "enabled": true,
    "mode": "inform",
    "inform_log_level": "info",
    "input": {
      "content_moderation": { "enabled": true },
      "prompt_injection": { "enabled": true },
      "pii": { "enabled": true }
    },
    "output": { "enabled": false }
  }
}
```

---

## 七、风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| OCI API 延迟增加 | 每次请求增加 ~100-500ms | 1. 并行调用 OCI 推理和 Guardrails 2. 可选跳过检查 |
| OCI API 不可用 | 请求失败 | 1. 添加超时和重试 2. 可选降级策略（跳过检查） |
| 误报 (False Positive) | 合法请求被拦截 | 1. 使用 inform 模式调试 2. 调整置信度阈值 |
| 漏报 (False Negative) | 有害内容通过 | 1. 定期更新过滤词 2. 多层检查 |
| 流式响应处理复杂 | 实现难度高 | 1. Phase 3 可选 2. 先实现缓冲后检查 |
| PII 仅支持英文 | 多语言场景受限 | 1. 文档说明 2. 等待 OCI 支持 |

---

## 八、验收标准

### 8.1 功能验收

- [ ] Input Guardrails 正常工作
  - [ ] Content Moderation 检测有害内容
  - [ ] Prompt Injection 检测注入攻击
  - [ ] PII 检测个人信息
  - [ ] 本地过滤词检查
- [ ] Output Guardrails 正常工作
  - [ ] Content Moderation 检查输出
  - [ ] PII 遮蔽功能
- [ ] Block 模式正确拦截违规请求
- [ ] Inform 模式正确记录日志
- [ ] 流式响应支持（可选）
- [ ] 配置文件正确加载

### 8.2 性能验收

- [ ] 单次 Guardrails 检查延迟 < 500ms
- [ ] 不影响非 Guardrails 请求性能

### 8.3 测试验收

- [ ] 单元测试覆盖率 > 80%
- [ ] 所有测试通过
- [ ] 手动测试场景通过

---

## 九、后续扩展（未来版本）

1. **自定义规则引擎**: 支持 `custom_rules.json` 中的正则规则
2. **热重载**: 不重启服务更新过滤词
3. **多语言 PII**: 等待 OCI 支持后启用
4. **指标与监控**: Prometheus metrics for guardrails
5. **审计日志**: 记录所有 guardrails 触发事件

---

## 十、总结

| 项目 | 内容 |
|------|------|
| **新建文件** | 6 个 |
| **修改文件** | 4 个 |
| **预估工作量** | 4 天 |
| **主要依赖** | OCI Generative AI SDK (已有) |
| **向后兼容** | ✅ 完全兼容（guardrails.enabled=false 时无影响） |

---

## 附录 A: OCI ApplyGuardrails API 参考

**API 端点**: `GenerativeAiInferenceClient.apply_guardrails()`

**请求结构**:
```python
apply_guardrails_details = oci.generative_ai_inference.models.ApplyGuardrailsDetails(
    input=oci.generative_ai_inference.models.GuardrailsTextInput(
        type="TEXT",
        content="要检查的文本内容",
        language_code="en"
    ),
    guardrail_configs=oci.generative_ai_inference.models.GuardrailConfigs(
        content_moderation_config=oci.generative_ai_inference.models.ContentModerationConfiguration(
            categories=["OVERALL", "BLOCKLIST"]
        ),
        personally_identifiable_information_config=oci.generative_ai_inference.models.PersonallyIdentifiableInformationConfiguration(
            types=["PERSON", "EMAIL", "TELEPHONE_NUMBER"]
        )
    ),
    compartment_id="ocid1.compartment.oc1..."
)
```

**响应结构**:
```json
{
  "results": {
    "contentModeration": {
      "categories": [
        { "name": "OVERALL", "score": 1.0 },
        { "name": "BLOCKLIST", "score": 0.0 }
      ]
    },
    "personallyIdentifiableInformation": [
      {
        "length": 15,
        "offset": 142,
        "text": "abc@example.com",
        "label": "EMAIL",
        "score": 0.95
      }
    ],
    "promptInjection": { "score": 1.0 }
  }
}
```

---

## 附录 B: 支持的 PII 类型

| 类型 | 描述 |
|------|------|
| `PERSON` | 人名 |
| `EMAIL` | 电子邮件地址 |
| `TELEPHONE_NUMBER` | 电话号码 |
| `CREDIT_CARD_NUMBER` | 信用卡号 |
| `IP_ADDRESS` | IP 地址 |
| `ADDRESS` | 物理地址 |
| `DATE_TIME` | 日期时间 |
| `LOCATION` | 地理位置 |

---

## 附录 C: 支持的语言 (Content Moderation & Prompt Injection)

- Arabic (Egyptian, Levantine, Saudi)
- BCMS (Bosnian, Croatian, Montenegrin, Serbian)
- Chinese (Standard Simplified, Standard Traditional)
- Dutch
- English
- French (France)
- German (Germany, Switzerland)
- Hebrew
- Hindi
- Indonesian
- Italian
- Japanese
- Korean
- Norwegian (Bokmål)
- Polish
- Portuguese (Brazilian, Portugal)
- Russian (Russia, Ukraine)
- Spanish (Spain)
- Swedish
- Thai
- Turkish
- Ukrainian
- Welsh

**注意**: PII 检测目前仅支持英文。
