import os
os.environ['TMPDIR'] = './temps' # avoid the system default temp folder not having access permissions
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # use huggingfacae mirror for users that could not login to huggingface

import re
import numpy as np
import matplotlib.pyplot as plt

import torch
import gradio as gr

from api import StableTTSAPI

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tts_model_path = './checkpoints/tsukuyomi_ft200.pt'
vocoder_model_path = './vocoders/pretrained/bigvgan_generator.pt'
vocoder_type = 'bigvgan'

model = StableTTSAPI(tts_model_path, vocoder_model_path, vocoder_type).to(device)

@ torch.inference_mode()
def inference(text, ref_audio, language, step, temperature, length_scale, solver, cfg, sway_coef, cfg_rescale, cfg_t_min, cfg_t_max, multi_ref, ref_window, extra_refs):
    text = remove_newlines_after_punctuation(text)

    if language == 'chinese':
        text = text.replace(' ', '')

    cfg_interval = (cfg_t_min, cfg_t_max) if cfg_t_min < cfg_t_max and (cfg_t_min, cfg_t_max) != (0.0, 1.0) else None
    refs = [ref_audio] + list(extra_refs or [])
    ref_arg = refs if len(refs) > 1 else ref_audio
    ref_window_seconds = ref_window if multi_ref else None

    audio, mel = model.inference(text, ref_arg, language, step, temperature, length_scale, solver, cfg, sway_coef=sway_coef, cfg_rescale=cfg_rescale, cfg_interval=cfg_interval, ref_window_seconds=ref_window_seconds)
    
    max_val = torch.max(torch.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    
    audio_output = (model.mel_config.sample_rate, (audio.cpu().squeeze(0).numpy() * 32767).astype(np.int16)) # (samplerate, int16 audio) for gr.Audio
    mel_output = plot_mel_spectrogram(mel.cpu().squeeze(0).numpy()) # get the plot of mel
    
    return audio_output, mel_output

def plot_mel_spectrogram(mel_spectrogram):
    plt.close() # prevent memory leak
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # remove white edges
    return fig

def remove_newlines_after_punctuation(text):
    pattern = r'([，。！？、“”‘’《》【】；：,.!?\'\"<>()\[\]{}])\n'
    return re.sub(pattern, r'\1', text)

def main():

    # gradio wabui, reference: https://huggingface.co/spaces/fishaudio/fish-speech-1
    gui_title = 'StableTTS'
    gui_description = """Next-generation TTS model using flow-matching and DiT, inspired by Stable Diffusion 3."""
    example_text = """こんにちは、今日はいい天気ですね！"""
    
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        demo.load(None, None, js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'light');window.location.search = params.toString();}}")

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"# {gui_title}")
                gr.Markdown(gui_description)

        with gr.Row():
            with gr.Column():
                input_text_gr = gr.Textbox(
                    label="Input Text",
                    info="Put your text here",
                    value=example_text,
                )
             
                ref_audio_gr = gr.Audio(
                    label="Reference Audio",
                    type="filepath"
                )

                with gr.Accordion('参照音声の詳細設定', open=False):
                    multi_ref_gr = gr.Checkbox(
                        label='参照を複数窓で平均',
                        value=False
                    )

                    ref_window_gr = gr.Slider(
                        label='窓長（秒）',
                        minimum=0.5,
                        maximum=5.0,
                        value=2.0,
                        step=0.5
                    )

                    extra_refs_gr = gr.Files(
                        label='追加参照音声（任意）',
                        file_count='multiple',
                        type='filepath'
                    )

                language_gr = gr.Dropdown(
                    label='Language',
                    choices=list(model.supported_languages),
                    value = 'japanese'
                )
                
                step_gr = gr.Slider(
                    label='Step',
                    minimum=1,
                    maximum=100,
                    value=16,
                    step=1
                )
                
                temperature_gr = gr.Slider(
                    label='Temperature',
                    minimum=0,
                    maximum=2,
                    value=1,
                )
                
                length_scale_gr = gr.Slider(
                    label='Length_Scale',
                    minimum=0,
                    maximum=5,
                    value=1,
                )
                
                solver_gr = gr.Dropdown(
                    label='ODE Solver',
                    choices=['euler', 'midpoint', 'dopri5', 'rk4', 'implicit_adams', 'bosh3', 'fehlberg2', 'adaptive_heun'],
                    value = 'euler'
                )

                sway_coef_gr = gr.Slider(
                    label='Sway Coef',
                    minimum=-1.0,
                    maximum=0.0,
                    value=-1.0,
                    step=0.05,
                    info='euler等の固定ステップソルバーでのみ有効'
                )

                cfg_gr = gr.Slider(
                    label='CFG',
                    minimum=0,
                    maximum=10,
                    value=3,
                )

                cfg_rescale_gr = gr.Slider(
                    label='CFG Rescale',
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    info='0=off。過剰CFGの飽和抑制'
                )

                cfg_t_min_gr = gr.Slider(
                    label='CFG t_min',
                    minimum=0.0,
                    maximum=0.9,
                    value=0.0,
                    step=0.05
                )

                cfg_t_max_gr = gr.Slider(
                    label='CFG t_max',
                    minimum=0.1,
                    maximum=1.0,
                    value=1.0,
                    step=0.05,
                    info='(0,1)以外でその区間のみguidance適用'
                )

            with gr.Column():
                mel_gr = gr.Plot(label="Mel Visual")
                audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
                tts_button = gr.Button("\U0001F3A7 Generate / 合成", elem_id="send-btn", visible=True, variant="primary")

        tts_button.click(inference, [input_text_gr, ref_audio_gr, language_gr, step_gr, temperature_gr, length_scale_gr, solver_gr, cfg_gr, sway_coef_gr, cfg_rescale_gr, cfg_t_min_gr, cfg_t_max_gr, multi_ref_gr, ref_window_gr, extra_refs_gr], outputs=[audio_gr, mel_gr])

    demo.queue()  
    demo.launch(debug=True, show_api=True,share=True)


if __name__ == '__main__':
    main()