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
    # 使用 awesome-digital-human-live2d 内置的免费角色模型
    LIVE2D_MODEL_PATH = r"C:\Users\b2655\Desktop\软件综合实训\workspace\awesome-digital-human-live2d\web\public\sentio\characters\free\Chitose\Chitose.model3.json"

    # ── 窗口 ──
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    FRAME_RATE = 60
