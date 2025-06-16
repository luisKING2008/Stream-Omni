# controller
python stream_omni/serve/controller.py --host 0.0.0.0 --port 10000

# CosyVoice worker
COSYVOICE_CKPT=path_to_CosyVoice-300M-25Hz # e.g., ./CosyVoice-300M-25Hz
WAV_DIR=path_to_save_generated_audio
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=CosyVoice/third_party/Matcha-TTS python ./CosyVoice/cosyvoice_worker.py --port 21003 --model ${COSYVOICE_CKPT} --wav_dir ./gen_wavs/

# Stream-Omni worker, add --load-8bit for VRAM lower than 32GB 
STREAMOMNI_CKPT=path_to_stream-omni-8b # e.g., ./stream-omni-8b
CUDA_VISIBLE_DEVICES=1  python ./stream_omni/serve/model_worker.py --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ${STREAMOMNI_CKPT} --model-name stream-omni

# Interface
python stream_omni/serve/gradio_web.py --controller http://localhost:10000 --model-list-mode reload  --port 7860
