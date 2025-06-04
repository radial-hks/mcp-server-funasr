# 基于FunASR的MCP服务器（MCPServer）

## 概述
MCPServer是一个基于Python的服务器，它借助阿里巴巴的FunASR库，通过FastMCP框架提供语音处理服务。它提供了以下工具：
 - **音频验证**：检查音频文件是否有效且可读，并提供其属性。
 - **语音转录**：使用Paraformer等先进的自动语音识别（ASR）模型，异步转录音频文件中的语音。支持管理转录任务并检索结果，包括详细的时间戳信息。
 - **语音活动检测（VAD）**：识别音频文件中的语音片段。

该服务器设计为可扩展的，并允许动态加载和切换ASR和VAD模型。

## 特性
 - **音频文件验证**：验证音频文件的完整性、可读性和格式。
 - **异步语音转文本转录**：非阻塞式转录，适用于长音频文件。
 - **转录任务管理**：启动任务、查询状态和检索结果。
 - **详细的转录结果**：可访问完整的转录文本、片段级别的开始/结束时间以及单词级别的时间戳（如果ASR模型提供）。
 - **语音活动检测（VAD）**：返回音频文件中语音片段的精确开始和结束时间戳。
 - **多模型支持**：利用FunASR多样化的模型库进行ASR和VAD。
 - **动态模型配置**：
    - 为每个转录或VAD请求指定模型。
    - 显式加载/切换服务器实例使用的默认ASR和VAD模型。
 - **可配置的模型参数**：将特定的加载和生成参数传递给FunASR模型。

## 前提条件
 - **Python**：3.8及以上版本。
 - **Pip**：用于安装Python包。
 - **MODELSCOPE_API_TOKEN（可选）**：
    - FunASR从ModelScope下载模型。如果你遇到速率限制或需要访问私有模型，可能需要设置`MODELSCOPE_API_TOKEN`环境变量。
    - 你可以从[ModelScope网站](https://www.modelscope.cn/)获取令牌。
    - 在环境中设置它：`export MODELSCOPE_API_TOKEN="YOUR_TOKEN_HERE"`

## 设置和安装
1. **克隆仓库（如果适用）**：
    如果此服务器是一个更大仓库的一部分，请克隆它。否则，请确保你有`MCPServer`目录及其内容。
2. **创建并激活虚拟环境（推荐）**：
```bash
python -m venv .venv
source .venv/bin/activate  # 在Windows上：.venv\Scripts\activate
```
3. **安装依赖项**：
    导航到`MCPServer`目录（包含此README和`server.py`的目录）。
    安装所需的软件包：
```bash
pip install -r requirements.txt
```
    这将安装`fastmcp`、`funasr`及其依赖项，包括PyTorch（如果FunASR的直接依赖项未另有指定，则默认为CPU版本）。如果你有特定的PyTorch需求（例如，GPU版本），建议在运行上述命令之前，按照[官方PyTorch网站](https://pytorch.org/get-started/locally/)的说明手动安装PyTorch。

## 运行服务器
1. **导航到`MCPServer`目录**。
2. **运行服务器应用程序**：
```bash
python server.py
```
3. 服务器将启动，你应该会看到日志输出，表明它正在运行，通常在`http://0.0.0.0:8000`上。首次运行时，FunASR将下载默认的ASR和VAD模型，这可能需要一些时间。

## 可用的MCP工具
你可以使用任何MCP客户端（例如，`mcp_client`或通过HTTP请求）与这些工具进行交互。服务器提供以下工具：

### 1. `validate_audio_file`
 - **描述**：验证音频文件是否适合处理，并提供其属性。
 - **参数**：
    - `file_path`（字符串，必填）：音频文件的路径。
 - **示例返回（成功）**：
```json
{
    "status": "valid",
    "message": "音频文件有效。",
    "details": {
        "samplerate": 16000,
        "channels": 1,
        "duration": 10.5,
        "formatted_duration": "00:10.500",
        "format": "WAV",
        "subtype": "PCM_16"
    }
}
```
 - **示例返回（错误 - 文件未找到）**：
```json
{
    "status": "invalid",
    "message": "错误：在'path/to/non_existent_audio.wav'未找到文件。",
    "details": null
}
```

### 2. `start_speech_transcription`
 - **描述**：为给定的音频文件启动异步语音转录任务。允许指定ASR模型和生成参数。
 - **参数**：
    - `audio_path`（字符串，必填）：音频文件的路径。
    - `model_name`（字符串，可选）：用于此任务的特定ASR模型（例如，ModelScope ID）。覆盖服务器当前的默认ASR模型。如果指定的模型尚未以兼容的设置加载，服务器将尝试使用该模型的默认加载参数或实例的通用默认加载参数加载它。
    - `model_generate_kwargs`（字典，可选）：ASR模型`generate`方法的特定参数（例如，`{"batch_size_s": 60, "hotword": "特定热词"}`）。这些参数将覆盖服务器中为当前ASR模型设置的任何默认生成参数。
 - **示例返回（成功）**：
```json
{
    "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "processing_started",
    "message": "转录任务已启动，正在处理中。"
}
```
 - **示例返回（错误 - 无效音频）**：
```json
{
    "task_id": null,
    "status": "error",
    "message": "错误：'/path/to/your/bad_audio.wav'处的文件不是有效的音频文件或已损坏。详细信息：<soundfile的错误内容>",
    "details": null
}
```
 - **示例返回（错误 - 任务期间模型加载失败）**：
```json
{
    "task_id": null,
    "status": "error",
    "message": "无法切换到模型'non_existent_model_id'。错误：加载模型'non_existent_model_id'时出错：<实际错误内容>"
}
```

### 3. `get_transcription_task_status`
 - **描述**：查询先前启动的语音转录任务的状态。
 - **参数**：
    - `task_id`（字符串，必填）：转录任务的唯一ID。
 - **示例返回（处理中）**：
```json
{
    "status": "processing",
    "audio_path": "/path/to/your/audio.wav",
    "model_used": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "submitted_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00",
    "details_from_validation": { /* ... 来自validate_audio的音频详细信息 ... */ },
    "model_generate_kwargs": {"batch_size_s": 300, "hotword": "魔搭"},
    "processing_started_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
}
```
 - **示例返回（已完成）**：
```json
{
    "status": "completed",
    "audio_path": "/path/to/your/audio.wav",
    "model_used": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "submitted_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00",
    "details_from_validation": { /* ... */ },
    "model_generate_kwargs": { /* ... */ },
    "processing_started_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00",
    "result": [ /* ... 实际转录结果 ... */ ],
    "completed_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
}
```
 - **示例返回（错误 - 任务未找到）**：
```json
{
    "status": "error",
    "message": "未找到任务ID。"
}
```

### 4. `get_transcription_result`
 - **描述**：检索已完成的语音转录任务的结果。
 - **参数**：
    - `task_id`（字符串，必填）：转录任务的唯一ID。
 - **示例返回（成功/已完成）**：
```json
{
    "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "completed",
    "result": [
        {
            "text": "这是 一段 测试 文本",
            "start": 120,
            "end": 2850,
            "timestamp": [[120, 300], [330, 500], [550, 900], [920, 1200]]
        }
    ],
    "completed_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
}
```
 - **示例返回（任务仍在处理中）**：
```json
{
    "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "processing",
    "message": "转录尚未完成或已失败。"
}
```
 - **示例返回（任务失败）**：
```json
{
    "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "failed",
    "message": "转录失败。",
    "error_details": "转录过程中的错误描述。",
    "failed_at": "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"
}
```

### 5. `load_asr_model`
 - **描述**：加载或重新加载特定的ASR模型，使其成为后续任务的默认模型，除非被覆盖。返回操作状态。
 - **参数**：
    - `model_name`（字符串，必填）：要加载的FunASR模型标识符（例如，ModelScope ID）。
    - `device`（字符串，可选）：加载模型的设备（例如，"cpu"，"cuda:0"）。如果为None，则使用实例默认值。
    - `model_load_kwargs`（字典，可选）：加载ASR模型的特定参数（例如，`{"ncpu": 2, "vad_model": "other-vad-id", "punc_model": "other-punc-id"}`）。这些参数将传递给`funasr.AutoModel`。
 - **示例返回（成功）**：
```json
{
    "status": "success",
    "message": "模型'iic/speech_paraformer-lar...（此处可能有截断，完整消息可能更详细）已成功加载。"
}
```