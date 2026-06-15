# 🌟 数字人情感陪伴系统

> 基于 SenseVoiceSmall + CosyVoice + LangGraph 的情感陪伴数字人系统
> 包含 **FastAPI 后端** + **live2d-py 桌面客户端**

---

## 📖 简介

**数字人情感陪伴系统**是一个多模态情感感知的 AI 陪伴系统，能够：

- 🎤 **语音识别** — 使用 SenseVoiceSmall 进行中英文语音识别 + 语音情绪感知
- 🗣️ **语音合成** — 使用 CosyVoice 生成自然语音，支持情绪驱动的语调调节
- 😊 **情绪感知** — 融合文本、语音、面部表情多模态情绪分析
- 🧠 **记忆系统** — 基于 Chroma 向量数据库的长期记忆，可搜索、加权
- 💬 **智能对话** — 基于 LangGraph 的工作流引擎，使用 DeepSeek/OpenAI LLM
- ⏰ **主动关怀** — 定时检查用户状态，低情绪时主动关怀
- 🖥️ **桌面客户端** — 基于 live2d-py 的原生窗口渲染，支持录音对话 + 口型同步

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     桌面客户端 (client/)                          │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 录音输入  │  │ HTTP/API  │  │ 语音播放  │  │ Live2D 渲染   │   │
│  │ pyaudio  │→│ requests  │←│ pygame   │  │ live2d-py    │   │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       │              │              │              │           │
│       │        POST /asr/upload     │              │           │
│       │        POST /chat           │              │           │
│       ▼              ▼              │              ▼           │
└────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────────┐
│                    FastAPI Server :8000                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ SenseVoice│  │ Fer(FER) │  │ Emotion  │  │ ActiveCare   │    │
│  │ ASR + SER  │  │ 面部情绪  │  │ Fusion   │  │ Scheduler    │    │
│  └─────┬─────┘  └─────┬─────┘  └────┬─────┘  └──────┬───────┘    │
│        │              │              │              │            │
│        ▼              ▼              ▼              ▼            │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │               LangGraph Agent Workflow                    │     │
│  │  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐   │     │
│  │  │Emotion  │→ │Memory   │→ │Prompt  │→ │LLM      │   │     │
│  │  │Analysis │  │Retrieve │  │Build   │  │Generate │   │     │
│  │  └─────────┘  └──────────┘  └────────┘  └──────────┘   │     │
│  │                                          │             │     │
│  │  ┌──────────┐  ┌──────────┐              │             │     │
│  │  │Store    │← │Response │←──────────────┘             │     │
│  │  │Memory   │  │Emotion  │                             │     │
│  │  └──────────┘  └──────────┘                             │     │
│  └──────────────────────────────────────────────────────────┘     │
│                    │              │                               │
│                    ▼              ▼                               │
│           ┌──────────┐    ┌──────────────┐                       │
│           │  CosyVoice│   │ Chroma Vector │                       │
│           │  TTS      │   │   DB (Memory) │                       │
│           └──────────┘    └──────────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Conda (推荐)
- Windows 10/11（桌面客户端需要 GPU + OpenGL）

### 安装后端依赖

```bash
# 1. 激活 conda 环境
conda activate rgzn

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 TTS 额外依赖
pip install matcha-tts pyworld

# 4. 设置环境变量（创建 .env 文件）
echo DEEPSEEK_API_KEY=your-api-key-here > .env
```

### 安装客户端额外依赖

```bash
pip install live2d-py glfw PyOpenGL pyaudio pygame pydub keyboard pyautogui
```

### 运行

**1. 启动后端**
```bash
# conda 环境
cd C:\Users\b2655\Desktop\软件综合实训\workspace
D:\anaconda\envs\rgzn\python.exe -m app.main
```
访问 `http://localhost:8000/docs` 查看 Swagger API 文档。

**2. 启动桌面客户端（新终端）**
```bash
conda activate rgzn
cd C:\Users\b2655\Desktop\软件综合实训\workspace
python -m client.main
```

**交互方式：** 按住 Ctrl 开始录音 → 说话 → 按住 Alt 结束 → 自动回复

---

## 🧩 模块说明

### 后端模块 (app/)

### 1. `app/models/schemas.py`
Pydantic 数据模型定义，包括情绪标签枚举、输入/输出模型、记忆条目等。

### 2. `app/asr/sensevoice.py`
语音识别模块。封装 FunASR 的 SenseVoiceSmall 模型，支持：
- 中英文语音转文字
- 语音情绪识别 (SER)
- 音频文件、字节流两种输入方式

### 3. `app/tts/cosyvoice_tts.py`
语音合成模块。封装 CosyVoice，支持：
- 自然语音生成 (inference_sft)
- 情绪驱动的语速/音调调节

### 4. `app/emotion/fer.py`
面部情绪识别模块。使用 FER 库检测面部表情，输出 7 类情绪。

### 5. `app/emotion/fusion.py`
多模态情绪融合模块。通过加权投票和 VETO 规则融合文本、语音、面部情绪。

### 6. `app/memory/rag.py`
RAG 记忆模块。基于 Chroma 向量数据库，支持：
- 语义搜索记忆
- 情感过滤
- 权重衰减
- 近期对话追溯

### 7. `app/agent/prompt.py`
提示词模板。包含系统提示（中文 "沐光" 人格设定）、主动关怀提示、情绪指导等。

### 8. `app/agent/workflow.py`
LangGraph 智能体工作流。**6 节点流水线**：
1. 分析情绪 → 2. 检索记忆 → 3. 构建提示 → 4. LLM 生成 → 5. 确定回复情绪 → 6. 存入记忆

### 9. `app/schedule/care.py`
主动关怀调度器。基于 APScheduler，定期检查用户状态：
- 长时间不活跃 → 打招呼
- 持续低情绪 → 安慰
- 中等不活跃 → 问候

### 10. `app/main.py`
FastAPI 主入口。包含所有 REST API 端点。

### 桌面客户端模块 (client/)

| 文件 | 说明 |
|------|------|
| `client/config.py` | 配置（后端地址、模型路径） |
| `client/api_client.py` | HTTP 客户端，调后端 ASR/Chat/TTS 端点 |
| `client/voice_input.py` | 录音模块（pyaudio + keyboard 快捷键） |
| `client/live2d_renderer.py` | Live2D 渲染 + 口型同步 + 眼动追踪 + 表情控制 |
| `client/main.py` | 主循环（录音→ASR→对话→TTS→播放） |

---

## 📦 依赖

请参见 [requirements.txt](requirements.txt) 和 [client/requirements.txt](client/requirements.txt) 完整列表。

### 后端核心依赖

| 包 | 用途 |
|---|---|
| `fastapi`, `uvicorn` | Web 框架 |
| `funasr`, `modelscope` | 语音识别 (SenseVoice) |
| `cosyvoice` | 语音合成（本地源码） |
| `matcha-tts`, `pyworld` | TTS 额外依赖 |
| `langgraph`, `langchain-deepseek` | AI Agent 工作流 |
| `chromadb`, `sentence-transformers` | 向量记忆 |
| `fer`, `opencv-python-headless` | 面部情绪识别 |
| `apscheduler` | 定时任务 |
| `torch`, `torchaudio` | 深度学习框架 |

### 客户端依赖

| 包 | 用途 |
|---|---|
| `live2d-py` | Python Live2D 渲染 |
| `glfw`, `PyOpenGL` | OpenGL 窗口 |
| `pyaudio` | 录音 |
| `pygame` | 音频播放 |
| `pydub` | 音频处理（口型同步） |
| `keyboard` | 全局快捷键 |
| `pyautogui` | 鼠标追踪 |
| `requests` | HTTP 客户端 |

---

## 🔑 环境变量

| 变量 | 说明 | 必需 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | ✅ |
| `OPENAI_API_KEY` | OpenAI API 密钥（备选） | ❌ |
| `PORT` | 服务端口（默认 8000） | ❌ |
| `HOST` | 服务主机（默认 0.0.0.0） | ❌ |

---

## 📁 项目结构

```
workspace/
├── app/                          # FastAPI 后端
│   ├── main.py                   # 主入口
│   ├── models/schemas.py         # 数据模型
│   ├── asr/sensevoice.py         # 语音识别 + SER
│   ├── tts/cosyvoice_tts.py      # 语音合成
│   ├── emotion/
│   │   ├── fer.py                # 面部情绪识别
│   │   └── fusion.py             # 多模态情绪融合
│   ├── memory/rag.py             # RAG 记忆
│   ├── agent/
│   │   ├── prompt.py             # 提示词模板
│   │   └── workflow.py           # LangGraph 工作流
│   └── schedule/care.py          # 主动关怀调度器
├── client/                       # Python 桌面客户端
│   ├── main.py                   # 客户端主入口
│   ├── config.py                 # 配置
│   ├── api_client.py             # HTTP 客户端
│   ├── voice_input.py            # 录音
│   ├── live2d_renderer.py        # Live2D 渲染 + 表情
│   └── requirements.txt          # 客户端依赖
├── CosyVoice/                    # CosyVoice 本地源码
├── pretrained_models/            # 预训练模型
├── docker/                       # Docker 配置
├── data/memory/                  # Chroma 持久化目录
├── requirements.txt              # 后端依赖
└── README.md
```

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — 多语言语音理解模型
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — 自然语音合成模型
- [LangGraph](https://github.com/langchain-ai/langgraph) — 图状态工作流框架
- [Chroma](https://github.com/chroma-core/chroma) — 向量数据库
- [live2d-py](https://github.com/Arkueid/live2d-py) — Python Live2D 渲染
- [suzuran0y/Live2D-LLM-Chat](https://github.com/suzuran0y/Live2D-LLM-Chat) — 客户端架构参考

---

> 🌈 用技术传递温暖，用陪伴治愈心灵
