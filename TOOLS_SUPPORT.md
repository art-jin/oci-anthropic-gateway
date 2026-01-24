# OCI-Anthropic Gateway - Enhanced Tool Support

## 概述 (Overview)

本文档说明了为解决OCI GenAI模型不能很好支持tools而进行的增强改进。OCI GenAI模型没有原生的function calling支持,经常把tool调用输出为文本而非结构化格式。

This document explains the enhancements made to solve the problem where OCI GenAI models don't support tools well. OCI GenAI models lack native function calling support and often output tool calls as text instead of structured format.

## 问题 (Problem)

OCI GenAI模型的局限性:
1. 没有原生的tool/function calling支持
2. 经常把tool调用变成普通文本输出
3. JSON格式不规范,缺少引号、逗号等
4. 有时包含额外的说明性文本

OCI GenAI model limitations:
1. No native tool/function calling support
2. Often outputs tool calls as plain text
3. Malformed JSON with missing quotes, commas, etc.
4. Sometimes includes explanatory text with tool calls

## 解决方案 (Solution)

### 1. 改进的Prompt工程 (Enhanced Prompt Engineering)

**文件**: `src/utils/tools.py` - `_build_tool_use_instruction()`

**改进内容**:
- 更详细的tool使用说明
- 明确的格式要求和示例
- 常见错误的避免指南
- 支持tool_choice策略 (auto/required/none/any)

**Improvements**:
- More detailed tool usage instructions
- Clear format requirements and examples
- Guidelines for avoiding common mistakes
- Support for tool_choice strategies (auto/required/none/any)

**示例Prompt格式**:
```
<TOOL_CALL>
{"name": "tool_name", "input": {"param": "value"}}
</TOOL_CALL>
```

### 2. 增强的JSON解析 (Enhanced JSON Parsing)

**文件**: `src/utils/json_helper.py` - `fix_json_issues()`

**新功能**:
- 自动修复缺少的引号
- 处理单引号替换为双引号
- 移除尾部逗号
- 补全不完整的JSON对象
- 修复未引用的属性名
- 提取嵌入在文本中的JSON对象

**New Features**:
- Auto-fix missing quotes
- Replace single quotes with double quotes
- Remove trailing commas
- Complete incomplete JSON objects
- Fix unquoted property names
- Extract JSON objects embedded in text

**修复示例**:
```javascript
// Before (Malformed)
{name: "search", input: {query: 'test',}}

// After (Fixed)
{"name": "search", "input": {"query": "test"}}
```

### 3. 工具调用检测 (Tool Call Detection)

**文件**: `src/utils/json_helper.py` - `detect_all_tool_call_blocks()`

**功能**:
- 使用正则表达式检测 `<TOOL_CALL>` 标签
- 支持不完整的结束标签
- 从文本中提取JSON
- 处理多个连续的tool调用
- 记录检测span位置以便从文本中移除

**Features**:
- Use regex to detect `<TOOL_CALL>` tags
- Support incomplete closing tags
- Extract JSON from text
- Handle multiple consecutive tool calls
- Track span positions for text removal

### 4. 自然语言回退机制 (Natural Language Fallback)

**文件**: `src/utils/json_helper.py` - `detect_natural_language_tool_calls()`

**用途**:
当模型不遵循精确的`<TOOL_CALL>`格式时的后备方案。

**Purpose**:
Fallback mechanism when models don't follow exact `<TOOL_CALL>` format.

**检测模式**:
1. 独立的JSON对象(带有"name"和"input"字段)
2. 自然语言表达模式:
   - "I'll use the [tool_name] tool..."
   - "Calling [tool_name] with..."
   - "Let me [tool_name]..."

**Detection Patterns**:
1. Standalone JSON objects with "name" and "input" fields
2. Natural language patterns:
   - "I'll use the [tool_name] tool..."
   - "Calling [tool_name] with..."
   - "Let me [tool_name]..."

### 5. 集成到生成服务 (Integration into Generation Service)

**文件**: `src/services/generation.py`

**流程**:
1. 首先检查结构化的tool调用(Cohere格式或Generic格式)
2. 如果没有,检查文本中的`<TOOL_CALL>`块
3. 如果仍然没有,使用自然语言回退机制
4. 从响应文本中移除tool调用标记
5. 正确格式化为Anthropic兼容的响应

**Flow**:
1. First check for structured tool calls (Cohere or Generic format)
2. If none, check for `<TOOL_CALL>` blocks in text
3. If still none, use natural language fallback
4. Remove tool call markers from response text
5. Format correctly as Anthropic-compatible response

## 使用示例 (Usage Examples)

### 示例1: 标准Tool调用

**用户请求**:
```json
{
  "model": "cohere.command-r-plus",
  "messages": [
    {"role": "user", "content": "搜索Python教程"}
  ],
  "tools": [
    {
      "name": "web_search",
      "description": "搜索网络信息",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "搜索查询"}
        },
        "required": ["query"]
      }
    }
  ],
  "tool_choice": "auto"
}
```

**模型输出** (可能格式不规范):
```
<TOOL_CALL>
{name: "web_search", input: {query: "Python tutorials"}}
</TOOL_CALL>
```

**网关处理后的输出**:
```json
{
  "id": "msg_123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "web_search",
      "input": {"query": "Python tutorials"}
    }
  ],
  "stop_reason": "tool_use"
}
```

### 示例2: 带说明文本的Tool调用

**模型输出**:
```
我会帮你搜索Python教程。
<TOOL_CALL>
{"name": "web_search", "input": {"query": "Python tutorials"}}
</TOOL_CALL>
让我知道是否需要其他帮助。
```

**网关处理后的输出**:
```json
{
  "content": [
    {
      "type": "text",
      "text": "我会帮你搜索Python教程。\n让我知道是否需要其他帮助。"
    },
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "web_search",
      "input": {"query": "Python tutorials"}
    }
  ],
  "stop_reason": "tool_use"
}
```

## 配置选项 (Configuration Options)

### tool_choice策略

- **auto** (默认): 需要时使用工具,简单问题直接回答
- **required**: 必须使用至少一个工具
- **any**: 适当时使用工具
- **none**: 不使用工具

### tool_choice strategies

- **auto** (default): Use tools when needed, answer directly for simple questions
- **required**: Must use at least one tool
- **any**: Use tools when appropriate
- **none**: Don't use tools

## 日志和调试 (Logging and Debugging)

网关会记录详细的调试信息:
- Tool调用检测结果
- JSON修复尝试
- 自然语言回退激活
- 解析失败的警告

The gateway logs detailed debug information:
- Tool call detection results
- JSON fix attempts
- Natural language fallback activation
- Parsing failure warnings

**查看日志**:
```bash
# 日志级别设置为INFO或DEBUG
export LOG_LEVEL=DEBUG
python main.py
```

## 限制和注意事项 (Limitations and Notes)

1. **Cohere模型**: 这些模型有更好的原生tool支持,增强功能主要用于Generic格式模型
2. **复杂工具**: 参数非常复杂的工具可能需要更明确的schema定义
3. **Token限制**: 大量的工具定义会占用更多prompt空间
4. **准确性**: 虽然已大幅改进,但某些边缘情况可能仍需要调整

1. **Cohere models**: These have better native tool support; enhancements mainly benefit Generic format models
2. **Complex tools**: Tools with very complex parameters may need more explicit schema definitions
3. **Token limits**: Large tool lists consume more prompt space
4. **Accuracy**: While greatly improved, some edge cases may still need adjustment

## 故障排除 (Troubleshooting)

### 问题: Tool调用未被检测到

**检查**:
1. 查看日志中的tool检测信息
2. 确认tool定义格式正确
3. 检查模型是否理解指令
4. 尝试简化tool描述

### Issue: Tool calls not detected

**Check**:
1. Review tool detection logs
2. Confirm tool definition format is correct
3. Check if model understands instructions
4. Try simplifying tool descriptions

### 问题: JSON解析失败

**检查**:
1. 查看原始模型输出日志
2. 检查是否有特殊字符
3. 确认JSON结构完整性
4. 查看fix_json_issues的尝试记录

### Issue: JSON parsing fails

**Check**:
1. Review raw model output logs
2. Check for special characters
3. Confirm JSON structure integrity
4. Review fix_json_issues attempt logs

## 性能考虑 (Performance Considerations)

- **正则表达式优化**: 使用预编译的正则表达式模式
- **增量处理**: 流式响应在接收完整文本后处理tool调用
- **缓存**: 考虑缓存tool定义转换结果

- **Regex optimization**: Uses pre-compiled regex patterns
- **Incremental processing**: Streaming responses process tool calls after receiving complete text
- **Caching**: Consider caching tool definition conversions

## 未来改进 (Future Improvements)

可能的增强方向:
1. 支持更多tool调用格式
2. 机器学习驱动的tool调用检测
3. 自适应prompt优化
4. Tool使用统计和分析

Possible enhancements:
1. Support for more tool call formats
2. ML-driven tool call detection
3. Adaptive prompt optimization
4. Tool usage statistics and analytics

## 贡献 (Contributing)

如果你发现改进空间或遇到特定的edge case,请提交issue或pull request。

If you find room for improvement or encounter specific edge cases, please submit an issue or pull request.

---

**最后更新** (Last Updated): 2026-01-24
**版本** (Version): 1.0
