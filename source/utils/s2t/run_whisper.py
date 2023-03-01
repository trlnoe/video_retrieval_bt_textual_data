import whisper

audio_path = '/workspace/competitions/AI_Challenge_2022/source/old_whisper/AIC_audio/C00_V0000.mp3'
model = whisper.load_model("base")
result = model.transcribe(audio_path)
print(result["text"])