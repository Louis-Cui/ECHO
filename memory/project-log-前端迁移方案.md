# 项目日志：前端迁移方案 — 代码详细设计

> 日期：2026-06-15 17:40  
> 基于 suzuran0y/Live2D-LLM-Chat 架构新建桌面客户端

---

## 一、文件结构

```
workspace/
├── app/                      ← 后端，不动
├── pretrained_models/        ← ASR/TTS 模型
├── awesome-digital-human-live2d/  ← 🗑️ 废弃（先备份）
├── client/                   ← 🆕 新建
│   ├── __init__.py
│   ├── config.py             ← 配置
│   ├── api_client.py         ← HTTP 调后端
│   ├── live2d_renderer.py    ← 从 suzuran0y 复制 + 改
│   ├── voice_input.py        ← 从 suzuran0y 复制 + 改
│   ├── main.py               ← 主循环
│   └── requirements.txt
├── temp/                     ← 临时录音/音频文件
└── ...
```

---

## 二、逐个文件代码

### 文件 1：`client/config.py`

```python
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # ── 后端 ──
    API_BASE = "http://127.0.0.1:8000"
    USER_ID = "live2d-desktop-001"

    # ── 录音 ──
    AUDIO_DIR = os.path.join(BASE_DIR, "..", "temp")
    AUDIO_INPUT = os.path.join(AUDIO_DIR, "input.wav")
    SAMPLE_RATE = 44100
    CHANNELS = 1
    CHUNK = 1024

    # ── Live2D 模型 ──
    # 指向你的模型文件（.model3.json）
    LIVE2D_MODEL_PATH = r"C:\Users\b2655\Desktop\软件综合实训\workspace\awesome-digital-human-live2d\web\public\models\your_model\your_model.model3.json"
    # 如果模型在别处，改这个路径

    # ── 窗口 ──
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    FRAME_RATE = 60
```

### 文件 2：`client/api_client.py`

```python
"""HTTP 客户端 → 后端 FastAPI"""

import base64
import requests
from config import Config

API = Config.API_BASE


def health() -> dict:
    """健康检查"""
    try:
        return requests.get(f"{API}/health", timeout=5).json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def asr(audio_path: str) -> dict:
    """
    语音识别
    返回示例: {"text": "你好", "emotion": {"emotion": "happy", "confidence": 0.85}}
    """
    with open(audio_path, "rb") as f:
        resp = requests.post(f"{API}/asr/upload", files={"file": f}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def chat(
    text: str,
    user_id: str = Config.USER_ID,
    emotion: dict = None,
) -> dict:
    """
    对话 + TTS
    返回示例: {
        "text": "今天过得怎么样？",
        "emotion": "happy",
        "tts": {"audio_base64": "xxx", "format": "wav", "duration_ms": 3000}
    }
    """
    body = {"text": text, "user_id": user_id}
    if emotion:
        body["emotion"] = emotion
    resp = requests.post(f"{API}/chat", json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()


def tts_only(text: str, emotion: str = "neutral") -> dict:
    """仅 TTS（不用完整对话流程时）"""
    resp = requests.post(
        f"{API}/tts",
        json={"text": text, "emotion": emotion},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def register_user() -> bool:
    """注册用户（主动关怀用）"""
    try:
        resp = requests.post(
            f"{API}/user/register",
            params={"user_id": Config.USER_ID},
            timeout=5,
        )
        return resp.ok
    except:
        return False
```

### 文件 3：`client/voice_input.py`

**来源：** 复制 suzuran0y 的 `ASR.py` 中的 `record_audio()` 方法

```python
"""
录音模块
来源: suzuran0y/Live2D-LLM-Chat/ASR.py — record_audio()
改动: 分离出独立的录音模块，去掉 ASR 推理部分
"""

import time
import wave
import pyaudio
import keyboard
from config import Config


class VoiceInput:
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.channels = Config.CHANNELS
        self.chunk = Config.CHUNK
        self.format = pyaudio.paInt16
        self.output_path = Config.AUDIO_INPUT

        import os
        os.makedirs(Config.AUDIO_DIR, exist_ok=True)

    def record(self) -> str:
        """
        录音：按住 Ctrl 开始，按住 Alt 结束
        返回: WAV 文件路径
        """
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        print("[按住 Ctrl 开始录音...]")
        keyboard.wait("ctrl")
        print("[录音中... 按住 Alt 结束]")

        frames = []
        while True:
            data = stream.read(self.chunk)
            frames.append(data)
            if keyboard.is_pressed("alt"):
                print("[录音结束]")
                break
            time.sleep(0.01)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # 保存 WAV
        with wave.open(self.output_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(p.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(frames))

        return self.output_path
```

### 文件 4：`client/live2d_renderer.py`

**来源：** 复制 suzuran0y 的 `Live2d_animation.py`

```python
"""
Live2D 渲染 + 口型同步 + 眼动追踪 + 表情控制
来源: suzuran0y/Live2D-LLM-Chat/Live2d_animation.py

改动清单（相比原版）:
  [+1] 新增 set_emotion() 方法 — 情绪参数映射
  [- ] play_live2d_once() — 原样保留
  [- ] update_gaze_tracking() — 原样保留
  [- ] extract_volume_array() — 原样保留
  [- ] play_audio_and_print_mouth() — 原样保留
  [- ] 窗口配置 — 原样保留
"""

import time
import threading
import glfw
import OpenGL.GL as gl
import pyautogui
import pygame
import ctypes
from pydub import AudioSegment
from live2d.v3 import LAppModel, init, dispose, glewInit, clearBuffer
from config import Config

# ── 窗口常量 ──
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020

# ── 眨眼状态 ──
BLINK_STATE_CLOSING = 1
BLINK_STATE_CLOSED = 2
BLINK_STATE_OPENING = 3

# ── 情绪→Live2D 参数映射（核心新增 ⭐） ──
# 格式: {情绪名: {参数ID: 值}}
# 这些参数是 Cubism 标准参数，如果模型没有某个参数，live2d-py 会静默跳过
EMOTION_PARAMS = {
    "happy": {
        "ParamBrowLY": -0.5,
        "ParamBrowRY": -0.5,
        "ParamMouthForm": 0.3,
        # ParamMouthOpenY 由 lip sync 驱动，这里不动
    },
    "love": {
        "ParamBrowLY": -0.3,
        "ParamBrowRY": -0.3,
        "ParamMouthForm": 0.2,
        "ParamEyeLOpen": 0.7,
        "ParamEyeROpen": 0.7,
    },
    "sad": {
        "ParamBrowLY": 0.5,
        "ParamBrowRY": 0.5,
        "ParamMouthForm": -0.5,
        "ParamEyeLOpen": 0.6,
        "ParamEyeROpen": 0.6,
    },
    "angry": {
        "ParamBrowLY": 0.8,
        "ParamBrowRY": 0.8,
        "ParamBrowLAngle": -0.5,
        "ParamBrowRAngle": 0.5,
        "ParamMouthForm": -0.8,
    },
    "surprised": {
        "ParamEyeLOpen": 1.5,
        "ParamEyeROpen": 1.5,
        "ParamBrowLY": -1.0,
        "ParamBrowRY": -1.0,
        "ParamMouthOpenY": 0.6,  # lip sync 会覆盖这个值，但初始是张开的
    },
    "fearful": {
        "ParamBrowLY": 0.6,
        "ParamBrowRY": 0.6,
        "ParamEyeLOpen": 1.3,
        "ParamEyeROpen": 1.3,
    },
    "anxious": {
        "ParamBrowLY": 0.4,
        "ParamBrowRY": 0.4,
        "ParamMouthForm": -0.3,
    },
    "upset": {
        "ParamBrowLY": 0.5,
        "ParamBrowRY": 0.5,
        "ParamMouthForm": -0.4,
        "ParamEyeLOpen": 0.7,
        "ParamEyeROpen": 0.7,
    },
}

# ── 重置默认值 ──
EMOTION_RESET = {
    "ParamEyeLOpen": 1.0,
    "ParamEyeROpen": 1.0,
    "ParamBrowLY": 0.0,
    "ParamBrowRY": 0.0,
    "ParamBrowLAngle": 0.0,
    "ParamBrowRAngle": 0.0,
    "ParamMouthForm": 0.0,
    # ParamMouthOpenY 由 lip sync 控制，不重置
}


class Live2DAnimationManager:
    """
    Live2D 动画管理器
    处理：模型加载、窗口渲染、眼动追踪、眨眼、口型同步、表情控制
    """

    def __init__(self, model_path, frame_rate=Config.FRAME_RATE):
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.mouth_value = 0.0  # 口型开合度（由 lip sync 设置）
        self.window = None
        self.model = None
        self.running = True

        # 眼动追踪
        self.last_mouse_x, self.last_mouse_y = pyautogui.position()
        self.last_move_time = time.time()
        self.IDLE_THRESHOLD = 3.0

        self.X_MIN, self.X_MAX = 200, 480
        self.Y_MIN, self.Y_MAX = 300, 360
        self.center_x_mapped = (self.X_MIN + self.X_MAX) / 2
        self.center_y_mapped = (self.Y_MIN + self.Y_MAX) / 2
        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.GAZE_EASING = 0.02

    # ═══════════════════════════════════════════════════════
    # 以下方法直接复制自 suzuran0y/Live2D-LLM-Chat/Live2d_animation.py
    # ═══════════════════════════════════════════════════════

    def configure_window(self, window, width, height):
        """配置 GLFW 透明窗口"""
        hwnd = glfw.get_win32_window(window)
        get_window_long = ctypes.windll.user32.GetWindowLongW
        set_window_long = ctypes.windll.user32.SetWindowLongW
        ex_style = get_window_long(hwnd, GWL_EXSTYLE)
        ex_style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
        set_window_long(hwnd, GWL_EXSTYLE, ex_style)

        glfw.make_context_current(window)
        screen_width, screen_height = pyautogui.size()
        glfw.set_window_pos(window, 0, screen_height - height)

    def load_live2d_model(self, width, height):
        """加载 Live2D 模型"""
        model = LAppModel()
        model.LoadModelJson(self.model_path)
        model.Resize(width, height)
        return model

    def play_live2d_once(self):
        """创建窗口并进入渲染循环"""
        init()
        if not glfw.init():
            print("GLFW 初始化失败！")
            return

        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.FLOATING, glfw.TRUE)

        window_width, window_height = Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT
        self.window = glfw.create_window(
            window_width, window_height, "Live2D Window", None, None
        )
        if not self.window:
            print("GLFW 窗口创建失败")
            glfw.terminate()
            return

        self.configure_window(self.window, window_width, window_height)
        glewInit()

        self.model = self.load_live2d_model(window_width, window_height)

        last_time = time.time()
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)

        while self.running and not glfw.window_should_close(self.window):
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            now = time.time()
            dt = now - last_time
            last_time = now

            width, height = glfw.get_framebuffer_size(self.window)
            gl.glViewport(0, 0, width, height)
            clearBuffer(0, 0, 0, 0)

            self.model.Update()
            # 口型同步（由外部 play_audio_and_print_mouth 设置 mouth_value）
            self.model.SetParameterValue("ParamMouthOpenY", self.mouth_value, 1.0)

            self.update_gaze_tracking(width, height)

            self.model.Draw()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        dispose()
        glfw.terminate()

    def update_gaze_tracking(self, width, height):
        """眼动追踪 — 鼠标跟随"""
        screen_x, screen_y = pyautogui.position()
        win_x, win_y = glfw.get_window_pos(self.window)
        local_mouse_x = screen_x - win_x
        local_mouse_y = screen_y - win_y

        if (screen_x != self.last_mouse_x) or (screen_y != self.last_mouse_y):
            self.last_move_time = time.time()
            self.last_mouse_x, self.last_mouse_y = screen_x, screen_y

        if (time.time() - self.last_move_time) < self.IDLE_THRESHOLD:
            mapped_x = self.X_MIN + (local_mouse_x / width) * (self.X_MAX - self.X_MIN)
            mapped_y = self.Y_MIN + (local_mouse_y / height) * (self.Y_MAX - self.Y_MIN)
            target_x = mapped_x
            target_y = mapped_y
        else:
            target_x = self.center_x_mapped
            target_y = self.center_y_mapped
            self.GAZE_EASING = 0.0004

        self.gaze_x += self.GAZE_EASING * (target_x - self.gaze_x)
        self.gaze_y += self.GAZE_EASING * (target_y - self.gaze_y)
        self.model.Drag(self.gaze_x, self.gaze_y)

    def extract_volume_array(self, audio_file):
        """
        提取音频音量数组 → 口型同步用
        返回: (volumes: list[float], duration_seconds: float)
        """
        seg = AudioSegment.from_file(audio_file, format="wav")
        frame_duration_ms = 1000.0 / self.frame_rate
        num_frames = int(seg.duration_seconds * self.frame_rate)

        volumes = []
        for i in range(num_frames):
            start_ms = i * frame_duration_ms
            frame_seg = seg[start_ms : start_ms + frame_duration_ms]
            rms = frame_seg.rms
            volumes.append(rms)

        max_rms = max(volumes) if volumes else 1
        volumes = [v / max_rms for v in volumes]
        return volumes, seg.duration_seconds

    def play_audio_and_print_mouth(self, audio_file):
        """
        播放音频 + 口型同步
        通过音量数组驱动 mouth_value，渲染循环读取它设置 ParamMouthOpenY
        """
        volume_array, audio_duration = self.extract_volume_array(audio_file)
        total_frames = len(volume_array)

        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        start_time = time.time()
        while True:
            current_time = time.time() - start_time
            if current_time >= audio_duration:
                break

            frame_index = int(current_time * self.frame_rate)
            if frame_index >= total_frames:
                frame_index = total_frames - 1

            self.mouth_value = volume_array[frame_index]

        pygame.mixer.music.stop()
        # 播完闭嘴
        self.mouth_value = 0.0

    # ═══════════════════════════════════════════════════════
    # 以下是新增方法（suzuran0y 原版没有）
    # ═══════════════════════════════════════════════════════

    def set_emotion(self, emotion: str):
        """
        设置模型情绪表情
        参数: emotion — "happy" / "sad" / "angry" / ...
        效果: 重置相关参数 → 应用情绪参数 → 嘴型参数不动（留给 lip sync）
        """
        if not self.model:
            return

        emotion = emotion.lower()

        # 重置参数
        for param_id, default_value in EMOTION_RESET.items():
            self.model.SetParameterValue(param_id, default_value, 1.0)

        # 应用情绪参数
        params = EMOTION_PARAMS.get(emotion, {})
        for param_id, value in params.items():
            self.model.SetParameterValue(param_id, value, 1.0)

        print(f"[表情] 设为 {emotion}")
```

### 文件 5：`client/main.py`

**来源：** 参考 suzuran0y 的 `main.py` 结构

```python
"""
数字人情感陪伴系统 — 桌面客户端主入口
基于 suzuran0y/Live2D-LLM-Chat 架构，后端调 workspace/app/ 的 FastAPI
"""

import base64
import os
import threading
import time

from config import Config
from api_client import asr, chat, health, register_user
from live2d_renderer import Live2DAnimationManager
from voice_input import VoiceInput


def main():
    print("=" * 50)
    print("数字人情感陪伴系统 — 桌面客户端")
    print("=" * 50)

    # ── 确保 temp 目录 ──
    os.makedirs(Config.AUDIO_DIR, exist_ok=True)

    # ── 检查后端 ──
    h = health()
    if h.get("status") != "ok":
        print(f"❌ 后端不可用: {h}")
        print("请先启动: python -m app.main")
        return
    print(f"✅ 后端健康: {h}")

    # ── 注册用户（主动关怀） ──
    register_user()
    print(f"✅ 用户已注册: {Config.USER_ID}")

    # ── 初始化 Live2D ──
    print(f"📦 加载模型: {Config.LIVE2D_MODEL_PATH}")
    live2d = Live2DAnimationManager(Config.LIVE2D_MODEL_PATH)
    voice = VoiceInput()

    # 启动渲染线程
    render_thread = threading.Thread(target=live2d.play_live2d_once, daemon=True)
    render_thread.start()
    time.sleep(1)  # 等窗口初始化

    # ── 主循环 ──
    print("\n🎤 按住 Ctrl 开始录音，按住 Alt 结束")
    print("   输入 'q' 回车退出\n")

    try:
        while True:
            # ── 1. 录音 ──
            wav_path = voice.record()

            # ── 2. ASR ──
            print("⏳ 语音识别中...")
            try:
                asr_result = asr(wav_path)
            except Exception as e:
                print(f"❌ ASR 失败: {e}")
                live2d.set_emotion("upset")
                continue

            user_text = asr_result.get("text", "").strip()
            user_emotion = asr_result.get("emotion", {})
            emotion_label = user_emotion.get("emotion", "neutral")

            if not user_text:
                print("⚠️ 未识别到文字")
                continue

            print(f"\n🧑 你说: {user_text}")
            print(f"😊 情绪: {emotion_label}")

            # ── 3. 设"倾听"表情 ──
            live2d.set_emotion(emotion_label)

            # ── 4. 对话 + TTS ──
            print("⏳ 思考中...")
            try:
                chat_result = chat(user_text, Config.USER_ID, user_emotion)
            except Exception as e:
                print(f"❌ 对话失败: {e}")
                live2d.set_emotion("neutral")
                continue

            reply_text = chat_result.get("text", "")
            reply_emotion = chat_result.get("emotion", "neutral")

            print(f"🤖 沐光: {reply_text}")
            print(f"😊 回复情绪: {reply_emotion}")

            # ── 5. 设"说话"表情 ──
            live2d.set_emotion(reply_emotion)

            # ── 6. 播放 TTS 音频 + 口型同步 ──
            tts_data = chat_result.get("tts", {})
            if tts_data and tts_data.get("audio_base64"):
                audio_bytes = base64.b64decode(tts_data["audio_base64"])
                tts_path = os.path.join(Config.AUDIO_DIR, "output.wav")
                with open(tts_path, "wb") as f:
                    f.write(audio_bytes)

                print("🔊 播放中...")
                live2d.play_audio_and_print_mouth(tts_path)

            # ── 7. 播完恢复 ──
            time.sleep(0.3)
            live2d.set_emotion("neutral")

    except KeyboardInterrupt:
        print("\n👋 退出")

    finally:
        live2d.running = False


if __name__ == "__main__":
    main()
```

### 文件 6：`client/requirements.txt`

```
# ── Live2D 渲染 ──
live2d-py>=0.2.0        # Python Live2D 渲染
glfw>=2.7.0             # OpenGL 窗口
PyOpenGL>=3.1.7         # OpenGL 绑定

# ── 音频 ──
pyaudio>=0.2.14         # 录音
pygame>=2.6.0           # 音频播放
pydub>=0.25.1           # 音频处理

# ── 快捷键 ──
keyboard>=0.13.5        # 全局快捷键

# ── HTTP ──
requests>=2.32.0        # HTTP 客户端
```

---

## 三、suzuran0y 原版 vs 我们版的对比

### `Live2d_animation.py` → `live2d_renderer.py`

```
suzuran0y 原版:
  set_mouth_value     -> 已有 (通过 mouth_value 属性)
  play_live2d_once    -> 已有
  update_gaze_tracking -> 已有
  extract_volume_array -> 已有
  play_audio_and_print_mouth -> 已有

我们新增:
  set_emotion(emotion)  -> 新增 ⭐
    -> EMOTION_RESET 重置参数
    -> EMOTION_PARAMS[emotion] 应用情绪参数
    -> 保留 ParamMouthOpenY 不动（让 lip sync 控制）
```

### `ASR.py` → `voice_input.py`

```
suzuran0y 原版:
  record_audio() + recognize_speech()
  (录音 + funasr 推理在同一个文件)

我们拆分:
  voice_input.py -> 只保留 record_audio() 录音部分
  api_client.py  -> ASR 推理改 HTTP 调用
```

### `main.py` → `main.py`

```
suzuran0y 原版:
  录音 → ASR(本地) → LLM(本地) → TTS(本地) → 播放 + 口型

我们版:
  录音 → api_client.asr() → 设倾听表情 → api_client.chat()
  → 设说话表情 → 播放 + 口型 → 恢复 neutral
```

---

## 四、后端对接验证

| 客户端调用 | 后端端点 | 后端处理 |
|-----------|---------|---------|
| `health()` | `GET /health` | `main.py` health() |
| `asr(path)` | `POST /asr/upload` | `main.py` speech_to_text_file() |
| `chat(text, uid, emotion)` | `POST /chat` | `main.py` chat() → LangGraph 工作流 |
| `register_user()` | `POST /user/register` | `main.py` register_user() |

所有后端端点已经在 `app/main.py` 中实现，**不需要改后端一行代码**。

---

## 五、实施步骤

| 步骤 | 操作 | 预计时间 |
|------|------|---------|
| 1 | 备份 awesome-digital-human-live2d 文件夹 | 2 min |
| 2 | 新建 `client/` 目录 + 写 6 个文件 | 30 min |
| 3 | 启动后端：`python -m app.main` | 1 min |
| 4 | 安装依赖：`pip install -r client/requirements.txt` | 5 min |
| 5 | 启动客户端：`python client/main.py` | 1 min |
| 6 | 按 Ctrl 录音，说一句话 | 5 min |
| 7 | 验证全链路 + 表情 | 10 min |

---

*2026-06-15 17:40 | 基于 suzuran0y 架构 + workspace 后端*
