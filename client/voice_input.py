# 录音模块
# 来源: suzuran0y/Live2D-LLM-Chat/ASR.py — record_audio()
# 改动: 分离出独立的录音模块，去掉 ASR 推理部分

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

    def record(self):
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
