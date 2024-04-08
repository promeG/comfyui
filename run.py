from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *

import sys

arguments = sys.argv

with Workflow():
    model, clip, _ = CheckpointLoaderSimple('cyberrealistic_v41BackToBasics.safetensors')
    model, clip = LoraLoader(model, clip, 'SDXLrender_v2.0.safetensors', 0.5, 0.5)
    upscale_model = UpscaleModelLoader('SwinIR_4x.pth')
    image, _ = LoadImage('input.jpg')
    image2 = ImageUpscaleWithModel(upscale_model, image)
    int, _ = MathExpressionPysssss('a.width // 5', image2, None, None)
    int2, _ = MathExpressionPysssss('a.height// 5', image2, None, None)
    model = TiledDiffusion(model, 'MultiDiffusion', int, int2, 32, 4)
    clip = CLIPSetLastLayer(clip, -2)
    string = WD14TaggerPysssss(image, 'wd-v1-4-moat-tagger-v2', 0.35, 0.85, False, False, '')
    conditioning = CLIPTextEncode(string, clip)
    conditioning2 = CLIPTextEncode('an image', clip)
    conditioning3 = ConditioningCombine(conditioning, conditioning2)
    control_net = ControlNetLoader('control_v11f1e_sd15_tile.pth')
    conditioning3 = ControlNetApply(conditioning3, control_net, image, 1)
    conditioning4 = CLIPTextEncode('embedding:BadDream,(UnrealisticDream:1.2),blurry,blurry background,blurry foreground,depth of field,motion blur,bokeh,', clip)
    vae = VAELoader('vae-ft-mse-840000-ema-pruned.safetensors')
    latent = VAEEncodeTiledTiledDiffusion(image2, vae, 3072, True, False)
    latent = KSampler(model, 580398480692331, 20, 5, 'dpmpp_2m', 'karras', conditioning3, conditioning4, latent, 0.45)
    image3 = VAEDecodeTiledTiledDiffusion(latent, vae, 384, True)
    SaveImage(image3, 'out')