export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=CosyVoice/third_party/Matcha-TTS

STREAMOMNI_CKPT=path_to_stream-omni-8b

# Replace the path of cosyvoice model in run_stream_omni.py (e.g., cosyvoice = CosyVoiceModel('./CosyVoice-300M-25Hz')) 
# add --load-8bit for VRAM lower than 32GB 
python ./stream_omni/eval/run_stream_omni.py \
    --model-path ${STREAMOMNI_CKPT} \
    --image-file ./stream_omni/serve/examples/cat.jpg --conv-mode stream_omni_llama_3_1 --model-name stream-omni  \
    --query ./stream_omni/serve/examples/cat_color.wav
