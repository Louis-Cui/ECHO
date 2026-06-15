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
from live2d.v3 import LAppModel, init, dispose, glInit, clearBuffer
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
        "ParamMouthOpenY": 0.6,
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
}


class Live2DAnimationManager:
    """
    Live2D 动画管理器
    处理：模型加载、窗口渲染、眼动追踪、眨眼、口型同步、表情控制
    """

    def __init__(self, model_path, frame_rate=Config.FRAME_RATE):
        self.model_path = model_path
        self.frame_rate = frame_rate
        self.mouth_value = 0.0
        self.window = None
        self.model = None
        self.running = True

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
        glInit()

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
        self.mouth_value = 0.0

    # ── 新增方法 ──

    def set_emotion(self, emotion):
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
