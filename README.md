# Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXX-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface%20-stream--omni--8b-orange.svg)](https://huggingface.co/ICTNLP/stream-omni-8b)
[![data](https://img.shields.io/badge/%F0%9F%93%91%20Datasets%20-InstructOmni-green.svg)](https://huggingface.co/datasets/ICTNLP/InstructOmni)
[![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fictnlp%2FStream-Omni&label=Visitors&icon=graph-up&color=%23dc3545)](https://github.com/ictnlp/Stream-Omni)

> [**Shaolei Zhang**](https://zhangshaolei1998.github.io/), [**Shoutao Guo**](https://scholar.google.com.hk/citations?user=XwHtPyAAAAAJ), [**Qingkai Fang**](https://fangqingkai.github.io/), [**Yan Zhou**](https://zhouyan19.github.io/zhouyan/), [**Yang Feng**](https://people.ucas.edu.cn/~yangfeng?language=en)\*


Stream-Omni is an end-to-end language-vision-speech chatbot that simultaneously supports interaction across various modality combinations, with the following features💡:
- **Omni Interaction**: Support any multimodal inputs including text, vision, and speech, and generate both text and speech responses.
- **Seamless "see-while-hear" Experience**: Simultaneously output *intermediate textual results* (e.g., ASR transcriptions and model responses) during speech interactions, like the advanced voice service of GPT-4o.
- **Efficient Training**: Require only a small amount of omni-modal data for training.

<p align="center" width="100%">
<img src="./assets/stream-omni.png" alt="stream-omni" style="width: 90%; min-width: 300px; display: block; margin: auto;">
</p>

## 🖥 Demo
| Microphone Input                                                | File Input                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <video src='https://github.com/user-attachments/assets/25807982-aa95-4633-9e92-10d995900258
' width="100%"/> | <video src='https://github.com/user-attachments/assets/df8d79ba-63db-487c-a4a9-f183372168a1
' width="100%"/> |

> [!NOTE]
>
> **Stream-Omni can produce intermediate textual results (ASR transcription and text response) during speech interaction, offering users a seamless "see-while-hear" experience.**




- Downlaod Stream-Omni model from [here](https://huggingface.co/ICTNLP/stream-omni-8b), put in `${STREAMOMNI_CKPT}`.
- Downlaod CosyVoice (Tokenizer & Flow Model) from [here](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz/files), put in `COSYVOICE_CKPT=./CosyVoice-300M-25Hz`:
    ```python
    from modelscope import snapshot_download
    snapshot_download('iic/CosyVoice-300M-25Hz', local_dir='./CosyVoice-300M-25Hz')
    ```
- Run these scripts to launch the API and interface, and then interact through the browser (http://localhost:7860):
    ```bash
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
    ```
- You can also refer to [`api.py`](./api.py) for the usage of API.

## 🔥 Quick Start

<p align="center" width="100%">
<img src="./assets/model.png" alt="model" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

> [!Tip]
>
> **Stream-Omni achieves modality alignments through sequence-dimension concatenation for vision-text alignment and layer-dimension mapping for speech-text alignment.**



### Requirements

- Install packages:
    ```bash
    conda create -n streamomni python=3.10 -y
    conda activate streamomni
    pip install -e .
    pip install flash-attn --no-build-isolation
    pip install -r requirements.txt
    pip install -r CosyVoice/requirements.txt
    ```
### Command Interaction
- Run these scripts for vision-grounded speech interaction:
    ```bash
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH=CosyVoice/third_party/Matcha-TTS
    
    STREAMOMNI_CKPT=path_to_stream-omni-8b
    
    # Replace the path of cosyvoice model in run_stream_omni.py (e.g., cosyvoice = CosyVoiceModel('./CosyVoice-300M-25Hz')) 
    # add --load-8bit for VRAM lower than 32GB 
    python ./stream_omni/eval/run_stream_omni.py \
        --model-path ${STREAMOMNI_CKPT} \
        --image-file ./stream_omni/serve/examples/cat.jpg --conv-mode stream_omni_llama_3_1 --model-name stream-omni  \
        --query ./stream_omni/serve/examples/cat_color.wav
    ```
    
    You should get the following outputs:
    
    ```yaml
    ASR Outputs:
    What is the color of the cat
    LLM Outputs:
    The cat is gray and black.
    Speech Tokens:
    <Audio_2164><Audio_2247><Audio_671><Audio_246><Audio_2172><Audio_1406><Audio_119><Audio_203><Audio_2858><Audio_2099><Audio_1716><Audio_22><Audio_1736><Audio_1038><Audio_4082><Audio_1655><Audio_2409><Audio_2104><Audio_571><Audio_2255><Audio_73><Audio_760><Audio_822><Audio_701><Audio_2583><Audio_1038><Audio_2203><Audio_1185><Audio_2103><Audio_1718><Audio_2610><Audio_1883><Audio_16><Audio_792><Audio_8><Audio_8><Audio_535><Audio_67>
    Speech Outputs:
    Audio saved at ./output_893af1597afe2551d76c37a75c813b16.wav
    ```
    
- Interaction across various modality combinations:

    | Inputs                    | Outputs | Intermediate Outputs                                         | Scripts                                                       |
    | ------------------------- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | Text + Vision (or None)   | Text    | /                                                            | [`run_stream_omni_t2t.py`](./stream_omni/eval/run_stream_omni_t2t.py) |
    | Text + Vision (or None)   | Speech  | Text result of model outputs                                 | [`run_stream_omni_t2s.py`](./stream_omni/eval/run_stream_omni_t2s.py) |
    | Speech + Vision (or None) | Text    | ASR transciption of user inputs                              | [`run_stream_omni_s2t.py`](./stream_omni/eval/run_stream_omni_s2t.py) |
    | Speech + Vision (or None) | Speech  | Text result of model outputs, ASR transciption of user inputs | [`run_stream_omni_s2s.py`](./stream_omni/eval/run_stream_omni_s2s.py) |

    > Control the interaction mode via `inference_type` in `model.generate()` (select from `text_to_text`, `text_to_speech`, `speech_to_text`, `speech_to_speech`)

### Evaluation
- Refer to [`./scripts/stream_omni/`](./scripts/stream_omni/) for evaluation scripts.

## 🤝 Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA)/[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)/[LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT): Stream-Omni is built upon the LLaVA and LLaVA-NeXT codebases and incorporates image instruction data from LLaVA-OneVision.
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice): Stream-Omni uses the tokenizer and flow model of CosyVoice.
- [UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio): Some normalization processing during evaluation refer to UltraEval-Audio.
- [VisIT-Bench](https://visit-bench.github.io/): Stream-Omni constructs SpokenVisIT benchmark based on VisIT-Bench for the evaluation of vision-grounded speech interaction.


## 🖋Citation

If this repository is useful for you, please cite as:

```

```

If you have any questions, please feel free to submit an issue or contact `zhangshaolei20z@ict.ac.cn`.
