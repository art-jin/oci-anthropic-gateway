# OCI-Anthropic Gateway Guardrails 最终现状说明

## 文档信息
- 日期: 2026-03-26
- 状态: Final current state
- 目标: 说明当前仓库中 OCI GenAI Guardrails 的已实现能力、配置结构、运行边界与验证状态

---

## 一、能力边界

当前实现严格区分两类能力：

1. **OCI-native Guardrails**
   - 由 OCI `ApplyGuardrails` 提供
   - 当前接入的 detector family：
     - Content Moderation
     - Prompt Injection
     - PII Detection

2. **Gateway-local Guardrails**
   - 由本项目在 gateway 内实现
   - 当前包括：
     - block / inform enforcement
     - block HTTP status / block message
     - log details / redact logs
     - input / output failure mode
     - local blocklist
     - whether to include `system` / `tool_result`
     - output PII rewrite
     - stream reject / downgrade behavior

这条边界已经在以下三处保持一致：

- 配置 schema
- 运行时逻辑
- 文档说明

---

## 二、当前支持范围

### 2.1 Input Guardrails

输入侧在请求进入 OCI 推理前执行。

当前支持：

1. OCI-native input content moderation
2. OCI-native input prompt injection detection
3. OCI-native input PII detection
4. gateway-local local blocklist
5. gateway-local whether to include `system`
6. gateway-local whether to include `tool_result`

当前输入检查对象：

1. 默认检查 `user` 文本
2. 可选检查 `system` 文本
3. 默认检查 `tool_result` 文本

当前不检查：

1. image 二进制内容
2. video 内容
3. `tool_use.input` 结构化 JSON 本身

### 2.2 Output Guardrails

输出侧仅在**非流式**响应中执行。

当前支持：

1. OCI-native output content moderation
2. OCI-native output PII detection
3. gateway-local output PII rewrite (`redact` / `mask`)

当前输出处理对象：

1. 仅处理 Anthropic response 中的 `text` block
2. 不修改 `tool_use`
3. 不修改 `metadata`
4. 不修改 `usage`
5. 不修改 `stop_reason`

### 2.3 Streaming 限制

当前不支持真正的流式 output guardrails 拦截。

当同时满足以下条件时：

1. `stream=true`
2. `guardrails.enabled=true`
3. `guardrails.oci_native.output.enabled=true`

当前行为由 `guardrails.gateway_extensions.streaming.when_output_guardrails_enabled` 控制：

1. `reject`
   - 返回 400
2. `downgrade_to_non_stream`
   - 走非流式路径
   - 返回普通 JSON
   - 增加 `metadata.guardrails_stream_downgraded=true`

---

## 三、代码接入点

### 3.1 输入侧

输入侧 Guardrails 接在：

- `src/routes/handlers.py`

当前顺序：

1. 请求参数校验
2. model routing / model config 解析
3. 输入文本收集
4. gateway-local blocklist + OCI-native input guardrails
5. 根据 `gateway_policy` 决定 block / inform / failure mode
6. 通过后继续 OCI chat inference

### 3.2 输出侧

输出侧 Guardrails 接在：

- `src/services/generation.py`

当前顺序：

1. OCI non-stream response 返回
2. Anthropic text blocks 提取
3. OCI-native output guardrails 执行
4. 根据 `gateway_policy` 决定 block / inform / failure mode
5. 若启用 gateway-local `pii_rewrite`，则仅重写 text blocks
6. 返回 Anthropic-compatible JSON

### 3.3 配置与解析

Guardrails 配置模型与 parser 位于：

- `src/utils/guardrails.py`

当前已实现：

1. dataclass schema
2. parser
3. enum validation
4. threshold validation
5. blocklist path resolution
6. OCI SDK request construction
7. result normalization / summary

---

## 四、当前配置结构

当前 `guardrails` schema：

```json
{
  "guardrails": {
    "enabled": false,
    "config_dir": "guardrails",
    "oci_native": {
      "default_language": "en",
      "input": {
        "enabled": true,
        "content_moderation": {
          "enabled": true,
          "categories": ["OVERALL"],
          "threshold": 0.5
        },
        "prompt_injection": {
          "enabled": true,
          "threshold": 0.95
        },
        "pii_detection": {
          "enabled": false,
          "types": ["EMAIL", "PERSON", "TELEPHONE_NUMBER"]
        }
      },
      "output": {
        "enabled": false,
        "content_moderation": {
          "enabled": true,
          "categories": ["OVERALL"],
          "threshold": 0.5
        },
        "pii_detection": {
          "enabled": false,
          "types": ["EMAIL", "PERSON", "TELEPHONE_NUMBER"]
        }
      }
    },
    "gateway_policy": {
      "mode": "block",
      "block_http_status": 400,
      "block_message": "Request blocked by gateway guardrails policy.",
      "log_details": false,
      "redact_logs": true,
      "input_failure_mode": "closed",
      "output_failure_mode": "open"
    },
    "gateway_extensions": {
      "input": {
        "include_system": false,
        "include_tool_results": true,
        "local_blocklist": {
          "enabled": false,
          "file": "blocklist.txt"
        }
      },
      "output": {
        "pii_rewrite": {
          "enabled": false,
          "action": "redact",
          "placeholder": "[REDACTED]"
        }
      },
      "streaming": {
        "when_output_guardrails_enabled": "reject"
      }
    }
  }
}
```

### 4.1 OCI-native 配置

- `guardrails.oci_native.default_language`
- `guardrails.oci_native.input.*`
- `guardrails.oci_native.output.*`

### 4.2 Gateway-local policy

- `guardrails.gateway_policy.mode`
- `guardrails.gateway_policy.block_http_status`
- `guardrails.gateway_policy.block_message`
- `guardrails.gateway_policy.log_details`
- `guardrails.gateway_policy.redact_logs`
- `guardrails.gateway_policy.input_failure_mode`
- `guardrails.gateway_policy.output_failure_mode`

### 4.3 Gateway-local extensions

- `guardrails.gateway_extensions.input.include_system`
- `guardrails.gateway_extensions.input.include_tool_results`
- `guardrails.gateway_extensions.input.local_blocklist.*`
- `guardrails.gateway_extensions.output.pii_rewrite.*`
- `guardrails.gateway_extensions.streaming.when_output_guardrails_enabled`

---

## 五、配置校验与默认行为

当前 parser 已校验：

1. `guardrails.gateway_policy.mode`
   - `block` / `inform`
2. `guardrails.gateway_policy.input_failure_mode`
   - `open` / `closed`
3. `guardrails.gateway_policy.output_failure_mode`
   - `open` / `closed`
4. `guardrails.gateway_extensions.streaming.when_output_guardrails_enabled`
   - `reject` / `downgrade_to_non_stream`
5. `guardrails.gateway_extensions.output.pii_rewrite.action`
   - `redact` / `mask`
6. `guardrails.gateway_policy.block_http_status`
   - 合法 HTTP 状态码
7. `guardrails.oci_native.*.*.threshold`
   - `0.0 <= threshold <= 1.0`

当前默认行为：

1. Input failure mode 默认 `closed`
2. Output failure mode 默认 `open`
3. Streaming + output guardrails 默认 `reject`
4. local blocklist 文件缺失时记录 warning，不阻止启动

---

## 六、运行时行为

### 6.1 Block / Inform

`guardrails.gateway_policy.mode` 当前支持：

1. `block`
   - 命中策略时直接返回安全错误响应
2. `inform`
   - 记录命中结果，但不阻断主流程

### 6.2 Failure mode

当前实现：

1. 输入侧异常
   - `input_failure_mode=open`：继续主流程
   - `input_failure_mode=closed`：直接拦截

2. 输出侧异常
   - `output_failure_mode=open`：继续返回原始输出
   - `output_failure_mode=closed`：直接拦截

### 6.3 PII Rewrite

当前实现：

1. OCI-native output PII detection 负责识别实体
2. gateway-local `pii_rewrite` 负责重写 Anthropic text block
3. 仅重写 text，不改 tool_use

---

## 七、测试与验证状态

### 7.1 单元测试

当前测试文件：

- `test/test_guardrails.py`

已覆盖：

1. 输入文本提取
2. local blocklist 匹配
3. PII redaction / masking
4. blocklist 路径解析
5. stream reject
6. stream downgrade
7. output redaction integration
8. OCI `TextContent` plain-text normalization
9. invalid config values
10. input failure mode open / closed
11. output failure mode open / closed

当前结果：

- `19 passed`

### 7.2 集成验证

当前已验证脚本：

1. `test/08_guardrails_input_block.py`
2. `test/09_guardrails_output_redact.py`
3. `test/10_guardrails_stream_downgrade.py`

已验证场景：

1. 输入阻断
2. 输出脱敏
3. 流式降级

---

## 八、当前限制

当前限制包括：

1. 不检查图片 / 视频内容
2. 不支持真正的实时流式 output guardrails 拦截
3. 不支持配置热重载
4. 不包含 Prometheus / 审计系统集成
5. 不对 `tool_use.input` 做结构化安全检查

---

## 九、后续方向

如继续扩展，优先方向为：

1. 更完整的 streaming downgrade 行为说明与回归验证
2. 多模态 Guardrails
3. 审计日志与 metrics
4. 配置热重载
5. 更细粒度的按模型 / 按路由策略

---

## 十、结论

当前仓库中的 Guardrails 已完成以下目标：

1. 配置、代码、文档三处严格区分 OCI-native 与 gateway-local
2. 输入与非流式输出 Guardrails 已接入主链路
3. 流式场景已实现 reject / downgrade 策略
4. failure mode、日志脱敏、PII rewrite 已落地
5. 相关单测与集成验证已通过

这份文档描述的是**当前已实现现状**，不再是历史计划稿。
