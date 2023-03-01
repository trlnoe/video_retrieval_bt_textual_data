audio_path='/workspace/competitions/AI_Challenge_2022/source/old_whisper/AIC_audio'

#python /workspace/competitions/AI_Challenge_2022/source/whisper/whisper/Mp3_convert.py
python transcribe_v2.py \
--audio $audio_path \
--model "large" \
--output_dir "/workspace/competitions/AI_Challenge_2022/source/whisper/AIC_S2T_text" 
#python /workspace/competitions/AI_Challenge_2022/source/whisper/whisper/Output_CSV.py