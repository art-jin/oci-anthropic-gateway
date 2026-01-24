# OCI-Anthropic 网关

English | [中文文档](README_CN.md)

一个生产就绪的转换层，使 OCI GenAI 模型能够无缝兼容 Anthropic 的 API 格式，包括全面的工具调用支持。

## 概述

该网关作为 Oracle Cloud Infrastructure (OCI) GenAI 服务与 Anthropic API 格式之间的桥梁，允许您：

1. **使用 OCI 托管的模型**（Grok、GPT 变体、Cohere Command-R 等）配合 Anthropic API 客户端
2. **启用工具调用**，为本身不支持该功能的模型通过高级提示工程实现
3. **流式传输响应**，采用 Anthropic 的服务器发送事件 (SSE) 格式
4. **访问高级功能**，如提示缓存、视觉、扩展思考等

**核心特性：**
- ✅ 完全兼容 Anthropic Messages API
- ✅ 增强的工具调用支持（原生 + 模拟）
- ✅ 流式和非流式响应
- ✅ 提示缓存支持
- ✅ 视觉/图像分析
- ✅ 模块化、可维护的代码库
- ✅ 生产就绪，包含全面的错误处理

## 快速开始

### 前置要求

- Python 3.12+
- 拥有 GenAI 服务访问权限的 OCI 账户
- 已配置 OCI CLI (`~/.oci/config`)

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd oci-anthropic-gateway

# 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

1. **复制配置模板：**
   ```bash
   cp config.json.template config.json
   ```

2. **编辑 `config.json`：**
   ```json
   {
     "compartment_id": "ocid1.compartment.oc1...",
     "endpoint": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
     "model_aliases": {
       "claude-3-5-sonnet-20241022": "gpt5",
       "claude-3-opus-20240229": "cohere.command-r-plus"
     },
     "model_definitions": {
       "gpt5": {
         "ocid": "ocid1.generativeaimodel.oc1...",
         "api_format": "generic",
         "max_tokens_key": "max_completion_tokens",
         "temperature": 1.0
       },
       "cohere.command-r-plus": {
         "ocid": "ocid1.generativeaimodel.oc1...",
         "api_format": "cohere",
         "max_tokens_key": "max_tokens",
         "temperature": 0.7
       }
     },
     "default_model": "gpt5"
   }
   ```

3. **配置 OCI SDK** (`~/.oci/config`)：
   ```ini
   [DEFAULT]
   user=ocid1.user.oc1...
   fingerprint=aa:bb:cc:dd...
   tenancy=ocid1.tenancy.oc1...
   region=us-chicago-1
   key_file=~/.oci/oci_api_key.pem
   ```

### 运行网关

```bash
# 使用模块化入口点（推荐）
python main.py

# 或使用传统单文件版本
python oci-anthropic-gateway.py
```

网关默认运行在 `0.0.0.0:8001`。

## 架构

代码库已从单文件 2000+ 行重构为模块化结构，以提高可维护性：

```
oci-anthropic-gateway/
├── main.py                      # 现代入口点
├── oci-anthropic-gateway.py     # 传统单文件版本（向后兼容）
├── config.json                  # 配置文件
├── src/
│   ├── config/                  # 配置管理
│   │   └── __init__.py         # Config 类，OCI 客户端初始化
│   ├── utils/                   # 工具模块
│   │   ├── constants.py        # 常量和停止原因
│   │   ├── token.py            # Token 计数工具
│   │   ├── tools.py            # 工具调用转换
│   │   ├── cache.py            # 缓存控制工具
│   │   ├── json_helper.py      # JSON 解析和修复
│   │   └── content_converter.py # 内容格式转换
│   ├── services/                # 业务逻辑
│   │   └── generation.py       # OCI 生成服务
│   └── routes/                  # API 路由
│       └── handlers.py         # 请求处理器
└── requirements.txt
```

### 核心组件

| 模块 | 用途 |
|--------|---------|
| **config/** | 加载配置、初始化 OCI 客户端 |
| **utils/constants.py** | 定义常量、停止原因、预编译正则表达式 |
| **utils/token.py** | Token 计数和估算 |
| **utils/tools.py** | 将 Anthropic 工具转换为 OCI/Cohere 格式，构建工具指令 |
| **utils/json_helper.py** | 解析和修复模型输出的畸形 JSON |
| **utils/content_converter.py** | 在 Anthropic、OCI 和 Cohere 格式之间转换 |
| **services/generation.py** | 核心生成逻辑（流式和非流式） |
| **routes/handlers.py** | FastAPI 路由处理器 |

## 工具调用支持

该网关的主要重点是为本机不支持函数调用的 OCI GenAI 模型提供强大的工具调用支持。

### 两种实现方式

#### 1. 原生函数调用（Cohere 模型）

配置为 `api_format: "cohere"` 的模型使用 OCI 的原生函数调用：

```json
{
  "model": "cohere.command-r-plus",
  "tools": [{
    "name": "get_weather",
    "description": "获取当前天气",
    "input_schema": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  }],
  "messages": [
    {"role": "user", "content": "东京天气怎么样？"}
  ]
}
```

#### 2. 模拟工具调用（通用模型）

对于 `api_format: "generic"` 的模型（xAI Grok、OpenAI GPT 等），网关使用高级提示工程：

**工作原理：**
1. 注入详细的系统提示，包含工具定义和使用说明
2. 模型以 `<TOOL_CALL>JSON</TOOL_CALL>` 格式输出工具调用
3. 网关检测、解析并转换为 Anthropic 格式
4. 支持一次响应中的多个工具调用

**模型输出示例：**
```
<TOOL_CALL>
{"name": "get_weather", "input": {"location": "东京"}}
</TOOL_CALL>
```

**网关转换为：**
```json
{
  "content": [{
    "type": "tool_use",
    "id": "toolu_abc123",
    "name": "get_weather",
    "input": {"location": "东京"}
  }],
  "stop_reason": "tool_use"
}
```

### 增强的 JSON 解析

网关包含复杂的 JSON 解析功能，用于处理模型输出的畸形数据：

**应用的修复：**
- 键/值周围缺少引号
- 单引号 → 双引号
- 尾随逗号
- 未加引号的属性名
- 不完整的 JSON 对象
- 文本中嵌入的 JSON

**示例：**
```javascript
// 模型输出（畸形）：
{name: "search", input: {query: 'test',}}

// 网关修复为：
{"name": "search", "input": {"query": "test"}}
```

### 自然语言回退

当模型不遵循确切的 `<TOOL_CALL>` 格式时，网关具有回退检测：

**检测的模式：**
- "我将使用 [tool_name] 工具..."
- "正在调用 [tool_name]，参数为..."
- "让我 [tool_name]..."
- 包含 "name" 和 "input" 字段的独立 JSON 对象

### 工具选择策略

```json
{
  "tool_choice": "auto"  // 选项：auto、required、any、none、{"type": "tool", "name": "..."}
}
```

| 策略 | 行为 |
|----------|----------|
| `auto`（默认） | 模型决定何时使用工具 |
| `required`/`any` | 必须使用至少一个工具 |
| `none` | 不使用工具 |
| `{"type": "tool", "name": "..."}` | 强制使用特定工具 |

### 工具结果格式

网关将工具结果转换为 Anthropic 格式，并使用清晰的类 XML 标记：

```xml
<TOOL_RESULT id='toolu_xxx' status='success'>
{"temperature": "22°C", "condition": "晴天"}
</TOOL_RESULT>
```

或错误情况：
```xml
<TOOL_RESULT id='toolu_xxx' status='error'>
错误信息在这里
</TOOL_RESULT>
```

## API 功能

### 支持的端点

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/v1/messages` | POST | 创建消息（流式和非流式） |
| `/v1/messages/count_tokens` | POST | 计算请求中的 token 数 |

### Messages API 功能

#### 1. 系统提示

```json
{
  "system": "你是一个有用的编程助手。",
  "messages": [...]
}
```

支持缓存的数组格式：
```json
{
  "system": [
    {"type": "text", "text": "你是一个有用的助手。"},
    {"type": "text", "text": "要简洁。", "cache_control": {"type": "ephemeral"}}
  ]
}
```

#### 2. 流式传输

```bash
curl -X POST http://localhost:8001/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "stream": true,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

发送兼容 Anthropic 的 SSE 事件：
- `message_start`
- `content_block_start`
- `content_block_delta`
- `content_block_stop`
- `message_delta`
- `message_stop`

#### 3. 视觉/图像

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "这张图片是什么？"},
      {
        "type": "image",
        "source": {
          "type": "base64",
          "media_type": "image/png",
          "data": "iVBORw0KGgo..."
        }
      }
    ]
  }]
}
```

#### 4. 扩展思考

```json
{
  "thinking": {
    "type": "enabled",
    "budget_tokens": 16000
  },
  "messages": [...]
}
```

#### 5. 提示缓存

```json
{
  "system": [{
    "type": "text",
    "text": "长系统提示...",
    "cache_control": {"type": "ephemeral"}
  }]
}
```

响应包含缓存指标：
```json
{
  "usage": {
    "input_tokens": 1500,
    "cache_creation_input_tokens": 1200,
    "cache_read_input_tokens": 300,
    "output_tokens": 200
  }
}
```

#### 6. 采样参数

```json
{
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "max_tokens": 4096,
  "stop_sequences": ["\n\n", "END"]
}
```

#### 7. 元数据

```json
{
  "metadata": {
    "user_id": "usr_12345",
    "conversation_id": "conv_abc789"
  }
}
```

在响应中回显。

#### 8. Token 计数

```bash
curl -X POST http://localhost:8001/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

响应：
```json
{
  "type": "usage",
  "input_tokens": 12
}
```

## 配置选项

### 模型配置

```json
{
  "model_definitions": {
    "model-key": {
      "ocid": "ocid1.generativeaimodel.oc1...",
      "api_format": "generic",  // 或 "cohere"
      "max_tokens_key": "max_completion_tokens",  // 或 "max_tokens"
      "temperature": 1.0
    }
  }
}
```

| 字段 | 描述 |
|-------|-------------|
| `ocid` | OCI 模型 OCID |
| `api_format` | `"generic"`（模拟工具）或 `"cohere"`（原生工具） |
| `max_tokens_key` | 参数名称：`"max_tokens"` 或 `"max_completion_tokens"` |
| `temperature` | 固定温度（覆盖请求） |

### 模型别名

将 Anthropic 模型名称映射到您的 OCI 模型：

```json
{
  "model_aliases": {
    "claude-3-5-sonnet-20241022": "gpt5",
    "claude-3-opus-20240229": "cohere.command-r-plus"
  }
}
```

## 日志和调试

网关提供灵活的日志记录，具有多个级别来控制详细程度。

### 日志级别

使用 `LOG_LEVEL` 环境变量控制日志输出：

```bash
# 最小日志记录（推荐用于生产环境）
LOG_LEVEL=WARNING python main.py

# 平衡日志记录（默认，推荐用于开发）
LOG_LEVEL=INFO python main.py

# 详细日志记录（用于调试问题）
LOG_LEVEL=DEBUG python main.py
```

### 记录的内容

**WARNING（最小）：**
- 配置已加载
- 服务器状态
- 仅关键错误

**INFO（默认 - 推荐）：**
- 请求摘要（流、工具计数）
- 工具调用检测结果
- 关键操作事件
- 空工具结果处理

**DEBUG（详细）：**
- 详细参数设置
- 内容转换详情
- JSON 解析尝试
- OCI SDK 内部请求

### 日志示例

**INFO 级别（清晰输出）：**
```
INFO:oci-gateway:Config loaded. Default model: xai.grok-code-fast-1
INFO:oci-gateway:OCI SDK initialized successfully
INFO:oci-gateway:Detected tool call: Bash with 1 parameters
INFO:oci-gateway:Tool detection result: found 1 tool calls
```

**DEBUG 级别（详细输出）：**
```
INFO:oci-gateway:Config loaded. Default model: xai.grok-code-fast-1
DEBUG:oci-gateway:Added system prompt (150 chars)
DEBUG:oci-gateway:Set top_p: 0.9
INFO:oci-gateway:Detected tool call: Bash with 1 parameters
DEBUG:oci-gateway:Raw JSON string: {"name": "Bash",...
```

### 主要特性

- **自动抑制 OCI SDK 日志**：在 INFO 和 WARNING 级别，隐藏详细的 OCI SDK 内部日志
- **环境变量控制**：无需更改代码，只需设置 `LOG_LEVEL`
- **清晰格式**：`LEVEL:logger_name:message`

完整的日志文档，请参阅 [LOGGING.md](LOGGING.md)。

## 故障排除

### 未检测到工具调用

**检查：**
1. 查看日志中的工具检测消息
2. 验证工具定义格式
3. 尝试在用户消息中使用明确的工具名称
4. 简化工具描述
5. 使用 `tool_choice: "required"` 测试

### JSON 解析错误

**检查：**
1. 查看日志中的原始模型输出
2. 检查特殊字符
3. 验证 JSON 结构
4. 查看 `fix_json_issues` 日志

### WebSearch 返回空

这是 Anthropic 客户端环境的问题，而非网关问题：
- 检查 API 密钥/凭据
- 验证网络访问
- 尝试不同的搜索查询
- 使用其他工具测试（Glob、Bash 等）

### 模型不遵循工具格式

**解决方案：**
1. 切换到 Cohere 模型以获得原生支持
2. 使用更明确的提示："使用 ListFiles 工具来..."
3. 设置 `tool_choice: "required"`
4. 检查模型兼容性

## 性能考虑

- **预编译正则**：在模块级别编译模式
- **流式效率**：在接收到完整文本后进行工具检测
- **Token 优化**：为 10+ 个工具简化指令
- **缓存**：考虑缓存工具定义转换

## 添加新模型

1. 从 OCI 控制台获取 OCI 模型 OCID
2. 添加到 `config.json` 中的 `model_definitions`：
   ```json
   {
     "my-model": {
       "ocid": "ocid1.generativeaimodel.oc1...",
       "api_format": "generic",  // 或 "cohere"
       "max_tokens_key": "max_tokens",
       "temperature": 0.7
     }
   }
   ```
3. 可选添加别名：
   ```json
   {
     "model_aliases": {
       "claude-3-5-sonnet-20241022": "my-model"
     }
   }
   ```

## 依赖项

```
fastapi>=0.104.0
uvicorn>=0.24.0
oci>=2.119.0
httpx>=0.25.0
python-dotenv>=1.0.0
```

## 安全注意事项

- `config.json` 在 `.gitignore` 中 - 永远不要提交它
- 使用 OCI IAM 策略限制模型访问
- 考虑为生产环境实现身份验证
- 监控 token 使用和成本

## 从单文件版本迁移

原始的 `oci-anthropic-gateway.py` 保持功能正常以实现向后兼容。要迁移：

1. **测试新版本：**
   ```bash
   python main.py
   ```

2. **更新任何脚本**引用旧文件

3. **如需要可继续使用旧版本** - 两者工作方式完全相同

## 未来增强

计划改进：
- [ ] 机器学习驱动的工具调用检测
- [ ] 自适应提示优化
- [ ] 工具使用统计和分析
- [ ] 增强的错误恢复
- [ ] 性能监控和追踪

## 贡献

欢迎贡献！请：
1. Fork 仓库
2. 创建功能分支
3. 为新功能添加测试
4. 提交拉取请求

## 许可证

本项目采用 Apache License 2.0 许可证。

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

译文说明：

根据 Apache 许可证 2.0 版本（以下简称"许可证"）授权，
除非遵守许可证，否则您不得使用此文件。

您可以在以下网址获取许可证副本：

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件
是按"原样"分发的，不附带任何明示或暗示的担保或条件。
有关许可证下特定语言管理权限和限制的详细信息，请参阅许可证。
```

## 支持

如有问题、疑问或贡献：
- 在 GitHub 上提出问题
- 查看日志以获取详细的错误信息
- 查阅 TOOLS_SUPPORT.md 了解工具调用详情
- 查阅 REFACTORING.md 了解架构详情

---

**最后更新**：2026-01-24
**版本**：2.0（模块化架构，增强工具支持）
