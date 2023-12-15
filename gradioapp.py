from diffusers import StableDiffusionPipeline
import torch

modelieo=['runwayml/stable-diffusion-v1-5',
 'dreamlike-art/dreamlike-diffusion-1.0',
 'CompVis/stable-diffusion-v1-4',
 'stabilityai/stable-diffusion-2-1',
 'nitrosocke/mo-di-diffusion',
 'prompthero/midjourney-v4-diffusion',
 'hakurei/waifu-diffusion',]


def T2I(Prompt,model):
  model_id = model
  pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
  pipe = pipe.to("cpu")

  prompt = Prompt
  image = pipe(prompt).images[0]

  return image


import gradio as gr


desc = 'The diffuser models are running on low processing power, which might take loger time to load the model...'
# css = ".gradio-container {background: rgb(0, 166, 228)}"
interface = gr.Interface(fn=T2I, 
                        inputs=["text", gr.Dropdown(modelieo)],
                         outputs="image", 
                        title='Advanced Stable DiFFusion',
                        article=desc
                        )

interface.launch()