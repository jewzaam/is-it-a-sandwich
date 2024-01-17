# AUTOGENERATED! DO NOT EDIT! File to edit: ../sandwich-serve.ipynb.

# %% auto 0
__all__ = ['learn', 'categories', 'image', 'label', 'examples', 'intf', 'classify_image']

# %% ../sandwich-serve.ipynb 4
from fastai.vision.all import *
import gradio as gr

# %% ../sandwich-serve.ipynb 5
learn = load_learner('sandwich.pkl')

# %% ../sandwich-serve.ipynb 6
categories = learn.dls.vocab

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

# %% ../sandwich-serve.ipynb 11
image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = list(map(str, get_image_files("example_data")))

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
