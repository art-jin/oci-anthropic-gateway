# OCI-Anthropic Gateway Guardrails 集成执行计划

## 文档版本
- 日期: 2026-03-12
- 状态: Revised for execution
- 目标: 在不破坏现有 Anthropic 兼容接口的前提下，为网关增加 OCI GenAI Guardrails 能力

---

## 一、目标与范围

### 1.1 本次版本目标

本次集成以"可落地、可验证、最小破坏"为原则，分阶段为网关增加以下能力：

1. Input Guardrails
   - 对进入 OCI 推理前的文本内容执行 Guardrails 检查
   - 支持 Content Moderation
   - 支持 Prompt Injection
   - 支持 PII 检测
   - 可选支持本地 blocklist 补充检查

2. Output Guardrails
   - 对 OCI 非流式响应文本执行 Guardrails 检查
   - 支持 Content Moderation
   - 支持 PII 检测
   - 支持对文本输出做 redact/mask

3. 运行模式
   - `block`: 命中策略时拦截输入，或阻断/替换输出
   - `inform`: 仅记录命中结果，不拦截

### 1.2 本次版本明确不做

以下内容不在 v1 范围内，避免把计划做成不可执行的大而全：

1. 不做图片/视频内容的 Guardrails 审核
   - 当前只对文本片段做检查
   - 多模态消息中的图片/视频内容不进入 Guardrails API

2. 不做真正的实时流式 Output Guardrails 拦截
   - 实时 SSE 一旦发送给客户端就无法撤回
   - v1 不实现"先发文本、结束后再告警"这种伪保护方案

3. 不做热重载
   - blocklist / pii 配置在进程启动时加载

4. 不做 Prometheus 指标与审计系统对接
   - 先保留结构化日志接口

---

## 二、基于当前代码的落地点

### 2.1 当前代码结构事实

现有项目的请求路径如下：

1. `src/routes/handlers.py`
   - 负责请求校验
   - 负责把 Anthropic messages 转成 OCI message
   - 负责分发到流式或非流式生成

2. `src/services/generation.py`
   - `generate_oci_non_stream()` 内部直接调用 OCI，并直接组装 Anthropic JSON 响应
   - `generate_oci_stream()` 内部直接输出 Anthropic SSE 事件流

因此：

1. Input Guardrails 应接在 `handlers.py` 中
   - 在参数校验通过后
   - 在调用 OCI 推理前

2. 非流式 Output Guardrails 应接在 `generate_oci_non_stream()` 中
   - 在拿到 OCI 响应并抽取出 `content_blocks` 后
   - 在返回 `JSONResponse` 前

3. 流式 Output Guardrails 不能沿用当前即时输出模型做真正拦截
   - v1 只定义清晰策略，不做半成品实现

### 2.2 关键兼容性约束

本计划执行时必须保证以下行为不被破坏：

1. `guardrails.enabled=false` 时，现有行为完全不变
2. 非流式接口仍返回 Anthropic 兼容 JSON
3. 流式接口默认仍返回 Anthropic 兼容 SSE
4. 工具调用 `tool_use` / `tool_result` 现有行为不被误改
5. `debug dump` 仍可用，但要补充脱敏策略

---

## 三、v1 策略定义

### 3.1 Input Guardrails 覆盖范围

v1 输入检查对象定义如下：

1. 默认检查 `user` 文本内容
2. 可选检查 `system` 文本内容
3. 默认检查 `tool_result` 文本内容
   - 原因: 当前项目会把 `tool_result` 以文本形式继续送给模型
   - 这是 prompt injection 的高风险入口，不能漏掉

不进入 v1 检查的内容：

1. image block 的二进制内容
2. video 内容
3. 纯结构化的 `tool_use.input` JSON 本身

### 3.2 Output Guardrails 覆盖范围

v1 输出检查对象定义如下：

1. 仅检查 Anthropic 响应中的 `text` content block
2. 不修改 `tool_use` block
3. 不修改 `metadata`
4. 不修改 `usage`
5. 不修改 `stop_reason`

### 3.3 流式响应策略

v1 对流式响应采用显式限制，而不是勉强支持：

1. Input Guardrails
   - 完全支持

2. Output Guardrails
   - `output.enabled=false`: 保持现有实时流式
   - `output.enabled=true` 且请求 `stream=true`:
     - 默认返回 400，提示当前配置不支持流式 output guardrails
     - 或由配置显式允许降级为非流式后再返回

建议 v1 默认采用第一种，逻辑清晰，风险最低。

### 3.4 失败策略

Guardrails API 自身也可能超时或失败，因此必须定义失败策略：

1. `fail_mode = "open"`
   - Guardrails 调用失败时跳过检查，继续主流程
   - 适合高可用优先

2. `fail_mode = "closed"`
   - Guardrails 调用失败时直接拦截请求
   - 适合合规优先

建议默认值：

1. Input Guardrails: `closed`
2. Output Guardrails: `open`

原因：

1. 输入侧保护优先级更高
2. 输出侧若因 Guardrails 故障导致全量失败，业务影响更大

---

## 四、配置设计

### 4.1 `config.json.template` 新增配置节

在现有配置中新增 `guardrails` 节：

```json
{
  "...": "...",
  "guardrails": {
    "enabled": false,
    "mode": "block",
    "default_language": "en",
    "config_dir": "guardrails",
    "block_http_status": 400,
    "block_message": "Request blocked by guardrails policy.",
    "streaming_behavior": "reject",
    "log_details": false,
    "redact_logs": true,
    "input": {
      "enabled": true,
      "fail_mode": "closed",
      "include_system": false,
      "include_tool_results": true,
      "content_moderation": {
        "enabled": true,
        "categories": ["OVERALL"]
      },
      "prompt_injection": {
        "enabled": true
      },
      "pii": {
        "enabled": false,
        "types": ["EMAIL", "PERSON", "TELEPHONE_NUMBER"]
      },
      "local_blocklist": {
        "enabled": false,
        "file": "blocklist.txt"
      }
    },
    "output": {
      "enabled": false,
      "fail_mode": "open",
      "content_moderation": {
        "enabled": true,
        "categories": ["OVERALL"]
      },
      "pii": {
        "enabled": false,
        "types": ["EMAIL", "PERSON", "TELEPHONE_NUMBER"],
        "action": "redact",
        "placeholder": "[REDACTED]"
      }
    }
  }
}
```

### 4.2 配置字段说明

| 字段 | 类型 | 建议默认值 | 说明 |
|------|------|------------|------|
| `enabled` | bool | false | Guardrails 总开关 |
| `mode` | string | `block` | `block` / `inform` |
| `default_language` | string | `en` | 传给 OCI Guardrails 的语言代码 |
| `config_dir` | string | `guardrails` | 外部模板/文件目录 |
| `block_http_status` | int | 400 | block 模式拦截时 HTTP 状态 |
| `block_message` | string | 固定文案 | 返回给客户端的安全文案 |
| `streaming_behavior` | string | `reject` | `reject` / `downgrade_to_non_stream`，v1 建议只支持 `reject` |
| `log_details` | bool | false | 是否记录 guardrails 细节 |
| `redact_logs` | bool | true | 日志中是否脱敏 |
| `input.enabled` | bool | true | 是否启用输入检查 |
| `input.fail_mode` | string | `closed` | `open` / `closed` |
| `input.include_system` | bool | false | 是否检查 system |
| `input.include_tool_results` | bool | true | 是否检查 tool_result |
| `output.enabled` | bool | false | 是否启用输出检查 |
| `output.fail_mode` | string | `open` | `open` / `closed` |

### 4.3 外部配置文件

v1 仅保留一个外部配置文件：

1. `guardrails/blocklist.txt.template`
   - 用于本地补充 blocklist
   - 运行时实际文件为 `guardrails/blocklist.txt`
   - 实际文件不提交仓库

PII 配置不单独放 `pii_config.json`，避免过度设计。v1 直接在 `config.json` 中配置类型列表和动作即可。

### 4.4 路径解析规则

必须明确，避免部署后路径错乱：

1. `config_dir` 相对 `config.json` 所在目录解析
2. `local_blocklist.file` 相对 `config_dir` 解析
3. 不依赖进程启动时的 cwd

### 4.5 配置校验规则

启动时应做以下校验：

1. `mode` 只能是 `block` / `inform`
2. `streaming_behavior` 只能是 `reject` / `downgrade_to_non_stream`
3. `fail_mode` 只能是 `open` / `closed`
4. `block_http_status` 必须是合法 HTTP 状态码
5. `pii.action` 只能是 `redact` / `mask` / `none`
6. `categories` 必须是数组
7. 若启用本地 blocklist 且文件不存在：
   - 记录 warning
   - 不自动创建目录和文件
   - 是否启动失败由配置决定，v1 默认不失败

---

## 五、文件变更计划

### 5.1 新建文件

| 文件路径 | 用途 |
|----------|------|
| `src/utils/guardrails.py` | Guardrails 核心封装 |
| `guardrails/blocklist.txt.template` | 本地补充 blocklist 模板 |
| `test/test_guardrails.py` | Guardrails 单元测试 |

### 5.2 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `config.json.template` | 新增 `guardrails` 配置节 |
| `src/config/__init__.py` | 解析并校验 Guardrails 配置 |
| `src/routes/handlers.py` | 接入 Input Guardrails 与流式限制策略 |
| `src/services/generation.py` | 接入非流式 Output Guardrails |
| `.gitignore` | 忽略运行时 blocklist 文件 |
| `README.md` | 补充配置和行为说明 |
| `README_CN.md` | 补充中文使用说明 |

说明：

1. 当前仓库测试目录为 `test/`，不是 `tests/`
2. 本次不新建 `guardrails/pii_config.json.template`

---

## 六、核心模块设计

### 6.1 新建 `src/utils/guardrails.py`

职责：

1. Guardrails 配置数据结构
2. 文本提取和归并
3. 调用 OCI `apply_guardrails`
4. 解析 OCI 返回结果
5. 本地 blocklist 检查
6. PII redaction / masking
7. 统一的结果对象与日志摘要

### 6.2 建议数据结构

建议按当前项目风格实现以下几类对象：

1. `GuardrailsConfig`
2. `InputGuardrailsConfig`
3. `OutputGuardrailsConfig`
4. `LocalBlocklistConfig`
5. `ContentModerationConfig`
6. `PromptInjectionConfig`
7. `PIIConfig`
8. `GuardrailsCheckResult`

### 6.3 关键函数

建议提供以下函数：

1. `collect_input_text_for_guardrails(body, config) -> str`
   - 从 Anthropic request body 提取需要检查的文本
   - 按 `include_system` / `include_tool_results` 决定范围

2. `apply_input_guardrails(...) -> GuardrailsCheckResult`
   - 执行 input 本地 + OCI 检查

3. `apply_output_guardrails(...) -> GuardrailsCheckResult`
   - 执行 output OCI 检查

4. `extract_text_blocks(content_blocks) -> str`
   - 从 Anthropic content blocks 中提取纯文本

5. `replace_text_blocks(content_blocks, new_text) -> list[dict]`
   - 仅替换 text block，不碰 tool_use block

6. `check_local_blocklist(text, blocklist) -> list[str]`

7. `load_blocklist(path) -> set[str]`

8. `redact_pii_text(text, pii_entities, action, placeholder) -> str`

9. `summarize_guardrails_result(result, redact_logs=True) -> dict`
   - 供日志和 debug dump 使用

### 6.4 OCI 调用约束

当前 SDK 调用是同步的，不能在 async 路径中直接阻塞 event loop。

因此：

1. `apply_guardrails` 必须通过 `asyncio.to_thread(...)` 执行
2. 必须传入 `compartment_id`
3. `compartment_id` 来源应优先使用当前请求最终选中的 model config

### 6.5 OCI 请求构造

根据 demo 和 SDK，本次计划使用：

1. `ApplyGuardrailsDetails`
2. `GuardrailsTextInput`
3. `GuardrailConfigs`
4. `ContentModerationConfiguration`
5. `PromptInjectionConfiguration`
6. `PersonallyIdentifiableInformationConfiguration`

---

## 七、配置加载改造计划

### 7.1 修改 `src/config/__init__.py`

新增内容：

1. Guardrails 相关配置属性
2. Guardrails 配置解析函数
3. Guardrails 配置校验函数
4. 相对路径解析逻辑

### 7.2 建议新增属性

```python
self.guardrails_enabled = False
self.guardrails = None
```

其余细分字段建议尽量收敛到 `self.guardrails` 对象内，不再平铺过多属性，避免 `Config` 继续膨胀。

### 7.3 设计原则

1. `Config` 负责读取 JSON 和基础校验
2. `src/utils/guardrails.py` 负责 Guardrails 子配置对象构建
3. 避免让 `src/config/__init__.py` 直接承载过多 Guardrails 业务逻辑

---

## 八、请求链路改造计划

### 8.1 修改 `src/routes/handlers.py`

#### 8.1.1 Input Guardrails 接入点

接入顺序：

1. 请求参数校验通过
2. 解析出 model config
3. 执行 Input Guardrails
4. 通过后继续消息转换与推理

原因：

1. Guardrails 需要用到最终模型的 `compartment_id`
2. 应在调用 OCI chat 前尽早拦截

#### 8.1.2 流式限制逻辑

当满足以下条件时：

1. `body.stream == true`
2. `guardrails.enabled == true`
3. `guardrails.output.enabled == true`

则：

1. 若 `streaming_behavior == "reject"`:
   - 返回 400
   - 响应说明当前配置不支持流式 output guardrails

2. 若 `streaming_behavior == "downgrade_to_non_stream"`:
   - v1 可选实现
   - 进入非流式生成路径
   - 最终仍返回普通 JSON，不返回 SSE

建议 v1 先只实现 `reject`。

### 8.2 修改 `src/services/generation.py`

#### 8.2.1 非流式输出接入点

在 `generate_oci_non_stream()` 中：

1. OCI chat 响应解析为 `content_blocks`
2. 若存在 output guardrails:
   - 仅提取 `text` block 组成文本
   - 调用 output guardrails
3. 根据模式处理：
   - `inform`: 只记录
   - `block`: 命中 content moderation 时返回安全错误
   - `block + pii redact/mask`: 替换 text block 后继续返回

#### 8.2.2 输出处理规则

1. 命中 output content moderation
   - `block` 模式下返回安全错误响应
   - `inform` 模式下记录命中并继续返回原始内容

2. 命中 output PII
   - 若配置 `action=none`:
     - 仅记录
   - 若 `action=redact` 或 `mask`:
     - 仅替换 `text` content block
     - 不改 `tool_use`

#### 8.2.3 流式函数本次不改主逻辑

`generate_oci_stream()` 在 v1 不做 output guardrails 半支持实现，避免：

1. 内容先发后审
2. SSE 协议破坏
3. 与工具调用检测缓冲逻辑相互干扰

---

## 九、错误响应设计

### 9.1 Input 拦截响应

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Request blocked by guardrails policy."
  }
}
```

### 9.2 Output 拦截响应

当输出内容命中强拦截规则时，非流式返回：

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Response blocked by guardrails policy."
  }
}
```

### 9.3 日志与 debug dump

v1 规则：

1. 默认不在客户端返回详细命中类别
2. 默认日志中不记录原始敏感文本
3. debug dump 中记录摘要，不记录未脱敏 PII 明文

---

## 十、测试计划

### 10.1 单元测试

文件：

1. `test/test_guardrails.py`

覆盖范围：

1. 输入文本提取
   - 只提取 user
   - 包含 system
   - 包含 tool_result
   - 忽略 image/video

2. blocklist
   - 空文件
   - 单条命中
   - 多条命中
   - 大小写
   - 注释和空行

3. PII 文本处理
   - redact
   - mask
   - 多实体
   - offset 重排

4. OCI 结果解析
   - content moderation 命中
   - prompt injection 命中
   - pii 命中
   - 空结果

5. 配置路径解析
   - 相对 `config.json` 路径
   - blocklist 缺失
   - 非法枚举值

### 10.2 服务级测试

建议新增 mock 级测试，验证：

1. Input Guardrails block 模式
2. Input Guardrails inform 模式
3. Guardrails API timeout + `fail_mode=open`
4. Guardrails API timeout + `fail_mode=closed`
5. Output text redact
6. Output moderation block
7. Output guardrails 开启时流式请求被拒绝

### 10.3 接口级回归测试

需补充或更新：

1. 非流式 messages 接口在 guardrails 关闭时行为不变
2. 流式 SSE 在 guardrails 关闭时行为不变
3. 开启 input guardrails 后违规输入被拦截
4. 开启 output guardrails 后非流式输出被替换或拦截

---

## 十一、文档更新计划

### 11.1 README.md / README_CN.md

补充内容：

1. `guardrails` 配置说明
2. v1 支持范围
3. 流式限制说明
4. 多模态限制说明
5. 失败策略说明

### 11.2 不单独创建 `GUARDRAILS.md`

v1 先把核心说明放在主 README 和中文 README 中，避免文档分叉。

---

## 十二、实施阶段

### Phase 1: 配置与框架

1. 更新 `config.json.template`
2. 新建 `guardrails/blocklist.txt.template`
3. 更新 `.gitignore`
4. 在 `src/config/__init__.py` 中解析 Guardrails 配置

交付结果：

1. 服务可识别 Guardrails 配置
2. Guardrails 关闭时行为不变

### Phase 2: 核心 Guardrails 模块

1. 新建 `src/utils/guardrails.py`
2. 实现 blocklist 加载
3. 实现输入文本抽取
4. 实现 OCI `apply_guardrails` 封装
5. 实现结果解析与日志摘要
6. 实现 PII redact/mask

交付结果：

1. 核心能力可独立单测

### Phase 3: Input 集成

1. 在 `src/routes/handlers.py` 接入 Input Guardrails
2. 增加 stream + output guardrails 的限制判断

交付结果：

1. 违规输入可被拦截或记录
2. 不支持的流式组合被显式拒绝

### Phase 4: Output 集成

1. 在 `src/services/generation.py` 接入非流式 Output Guardrails
2. 实现 text block 替换
3. 实现安全错误响应

交付结果：

1. 非流式输出支持 redact / moderation block

### Phase 5: 测试与文档

1. 完成单元测试
2. 完成 mock 服务级测试
3. 更新 README.md / README_CN.md
4. 手动验证关键场景

---

## 十三、验收标准

### 13.1 功能验收

- [ ] `guardrails.enabled=false` 时，现有接口行为不变
- [ ] Input Guardrails 可检测 content moderation
- [ ] Input Guardrails 可检测 prompt injection
- [ ] Input Guardrails 可检测 PII
- [ ] 本地 blocklist 可生效
- [ ] 非流式 Output Guardrails 可执行 content moderation
- [ ] 非流式 Output Guardrails 可执行 PII redact/mask
- [ ] `inform` 模式只记录不拦截
- [ ] `block` 模式按策略拦截
- [ ] 流式 + output guardrails 的组合被显式拒绝或按配置降级

### 13.2 兼容性验收

- [ ] 非流式 Anthropic JSON 结构不变
- [ ] 流式 SSE 在 guardrails 关闭时结构不变
- [ ] 现有工具调用逻辑不被破坏
- [ ] 多模型路由逻辑不被破坏

### 13.3 测试验收

- [ ] 新增 Guardrails 单元测试通过
- [ ] 现有测试无回归
- [ ] 至少覆盖 block / inform / fail_open / fail_closed 四类核心路径

---

## 十四、风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|----------|
| Guardrails API 增加额外延迟 | 请求耗时上升 | v1 先接受串行开销，后续再优化 |
| Guardrails API 故障 | 影响请求成功率 | 配置 `fail_mode`，按输入/输出区分策略 |
| 日志记录敏感内容 | 合规风险 | 默认 `redact_logs=true`，日志只记摘要 |
| 流式输出无法真正拦截 | 保护不完整 | v1 明确拒绝该组合，不做半支持 |
| 多模态请求只检查文本 | 有覆盖盲区 | README 明确限制，后续版本补齐 |
| tool_result 中存在注入文本 | 输入侧绕过 | 默认纳入 input guardrails 检查 |

---

## 十五、后续版本路线

以下能力留待 v2 及以后：

1. `streaming_behavior=downgrade_to_non_stream` 的完整实现
2. 审计日志与 metrics
3. 配置热重载
4. 多模态 Guardrails
5. 更细粒度的按模型/按路由 Guardrails 策略

---

## 十六、结论

这版执行计划相对旧计划做了以下关键修正：

1. 明确了真正的代码接入点
   - Input 在 `handlers.py`
   - 非流式 Output 在 `generation.py`

2. 删除了不可落地的流式 output guardrails 伪方案

3. 把 `tool_result` 纳入输入检查范围

4. 增加了失败策略、路径解析、日志脱敏和兼容性约束

5. 把测试计划改成与当前仓库实际结构一致的 `test/` 目录方案

该计划可以直接作为后续实现基线使用。
