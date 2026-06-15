"""
数字人情感陪伴系统 — 桌面客户端主入口
基于 suzuran0y/Live2D-LLM-Chat 架构，后端调 workspace/app/ 的 FastAPI
"""

import base64
import os
import sys
import threading
import time

# 确保 client/ 目录在 Python 路径中
_client_dir = os.path.dirname(os.path.abspath(__file__))
if _client_dir not in sys.path:
    sys.path.insert(0, _client_dir)

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
