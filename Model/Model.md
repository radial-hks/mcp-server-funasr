# Model 目录结构

**[模型的安装与管理可以完全参照官方文档完成。](https://www.modelscope.cn/docs/models/download)**


## 根目录文件
- `asr_test.py` - ASR测试脚本
- `Model.md` - 本文档

## speech_fsmn_vad_zh-cn-16k-common-pytorch
语音识别模型目录，包含以下文件：
- `model.pt` - 模型文件 (1.6MB)
- `configuration.json` - 配置文件
- `am.mvn` - 模型均值方差文件
- `config.yaml` - 模型配置
- `README.md` - 说明文档
- `.gitattributes` - Git属性配置

子目录：
- `fig/` - 图片资源目录
- `example/` - 示例文件目录
- `.git/` - Git版本控制目录

## SenseVoiceSmall
多语言语音识别模型目录，包含以下文件：
- `model.pt` - 模型文件 (893MB)
- `chn_jpn_yue_eng_ko_spectok.bpe.model` - BPE模型文件
- `tokens.json` - 词表文件
- `configuration.json` - 配置文件
- `config.yaml` - 模型配置
- `am.mvn` - 模型均值方差文件
- `.gitattributes` - Git属性配置
- `README.md` - 说明文档

子目录：
- `fig/` - 图片资源目录
- `example/` - 示例文件目录
- `.git/` - Git版本控制目录

