# AI 集成使用说明

## 概述

股票分析系统现已支持多种AI提供商进行投资建议分析：
- **DeepSeek**（默认）
- **智谱AI**

## 配置说明

### 1. DeepSeek 配置

DeepSeek 是当前的**默认AI提供商**。

#### 环境变量设置

需要设置环境变量 `DEEPSEEK_API_KEY`：

**Windows (PowerShell):**
```powershell
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key-here"
```

**Windows (CMD):**
```cmd
set DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

**Linux/Mac:**
```bash
export DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

#### 配置文件
配置位于 `config.py` 中的 `AI_CONFIG['deepseek']`：
```python
'deepseek': {
    'api_key': os.environ.get('DEEPSEEK_API_KEY', ''),
    'base_url': 'https://api.deepseek.com',
    'model': 'deepseek-chat',
    'max_tokens': 100,
    'temperature': 0.6
}
```

### 2. 智谱AI 配置

配置位于 `config.py` 中的 `AI_CONFIG['zhipuai']`，API密钥已经内置在配置文件中。

### 3. 切换AI提供商

在 `config.py` 的 `AI_CONFIG` 中修改 `default_provider`：
```python
'default_provider': 'deepseek',  # 或 'zhipuai'
```

## 使用示例

### 基础使用（使用默认提供商）

```python
from stock_analyzer.ai_client import get_ai_analyzer

# 获取AI分析器（使用默认配置中的DeepSeek）
analyzer = get_ai_analyzer()

# 生成投资建议
result = analyzer.analyze_investment_recommendation(
    technical_report="技术分析报告内容...",
    stock_code="600000"
)

# 结果解释：
# 1: 立即买入
# -1: 不适合买入
# 0: 观望等待
```

### 指定AI提供商

```python
from stock_analyzer.ai_client import get_ai_analyzer

# 使用 DeepSeek
analyzer = get_ai_analyzer(provider='deepseek')

# 使用 智谱AI
analyzer = get_ai_analyzer(provider='zhipuai')
```

### 手动指定API密钥

```python
from stock_analyzer.ai_client import AIStockAnalyzer

# 创建DeepSeek分析器
analyzer = AIStockAnalyzer(
    api_key="your-deepseek-api-key",
    provider='deepseek'
)

# 创建智谱AI分析器
analyzer = AIStockAnalyzer(
    api_key="your-zhipuai-api-key",
    provider='zhipuai'
)
```

### 检查AI功能是否可用

```python
analyzer = get_ai_analyzer()

if analyzer.is_available():
    print("AI功能可用")
else:
    print("AI功能不可用，请检查配置")
```

## API接口对比

### DeepSeek API
```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ],
    stream=False
)
```

### 智谱AI API
```python
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="glm-4.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ]
)
```

## 投资建议返回值说明

| 返回值 | 含义 | 说明 |
|--------|------|------|
| 1 | 立即买入 | 技术面良好，建议立即买入 |
| 0 | 观望等待 | 当前不是最佳买点，建议观望 |
| -1 | 不适合买入 | 风险高于收益，不建议买入 |

## 故障排查

### 1. AI功能不可用

**问题**：提示"AI功能不可用"

**解决方案**：
- 检查是否设置了环境变量 `DEEPSEEK_API_KEY`
- 确认已安装 `openai` 包：`pip install openai`
- 查看日志文件了解详细错误信息

### 2. DeepSeek API密钥未设置

**问题**：提示"未提供DeepSeek API密钥"

**解决方案**：
```bash
# 设置环境变量
export DEEPSEEK_API_KEY=your-api-key-here
```

### 3. 网络连接问题

**问题**：API请求超时或失败

**解决方案**：
- 检查网络连接
- 确认API服务可访问
- 查看日志文件获取详细错误信息

## 日志查看

日志文件位于项目根目录的 `logs/` 文件夹：
- `stock_analyzer.log` - 所有日志
- `stock_analyzer_errors.log` - 错误日志

## 注意事项

1. **API密钥安全**：请勿将API密钥硬编码在代码中或提交到版本控制系统
2. **Token限制**：注意API调用的token使用量，避免超出配额
3. **成本控制**：根据实际需求调整 `max_tokens` 参数以控制成本
4. **错误处理**：当AI功能不可用时，系统会自动返回观望建议（0）

## 更新历史

- **2025-10-10**: 添加 DeepSeek 支持，设置为默认AI提供商
- 支持多AI提供商切换
- 优化错误处理和日志记录

