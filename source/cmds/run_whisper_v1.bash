cd /workspace/competitions/AI_Challenge_2022/source/whisper/whisper
audio_path='/workspace/competitions/AI_Challenge_2022/source/old_whisper/AIC_audio/C00_V0000.mp3'
python transcribe.py \
--audio $audio_path \
--model "small" \
--output_dir "/workspace/competitions/AI_Challenge_2022/source/whisper/output" 
#python /workspace/competitions/AI_Challenge_2022/source/whisper/whisper/Output_CSV.py