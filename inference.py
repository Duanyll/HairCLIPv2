import os
# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

print(">>> Importing modules...")

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scripts.Embedding import Embedding
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy
# from scripts.sketch_proxy import SketchProxy
from scripts.bald_proxy import BaldProxy
from scripts.color_proxy import ColorProxy
from scripts.feature_blending import hairstyle_feature_blending
# from utils.seg_utils import vis_seg
# from utils.mask_ui import painting_mask
from utils.image_utils import display_image_list, process_display_input
print(">>> Importing base modules...")
from utils.model_utils import load_base_models
print(">>> Base modules imported")
from utils.options import Options

from utils.latent_cache import LatentCache

print(">>> Modules imported")

class InferenceProxy:
    def __init__(self) -> None:
        self.opts = Options().parse()
        self.image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        print(">>> Loading models...")
        self.g_ema, self.mean_latent_code, self.seg = load_base_models(self.opts)
        print(">>> Models loaded")
        self.ii2s = Embedding(self.opts, self.g_ema, self.mean_latent_code[0,0])
        print(">>> Embedding loaded")

        self.latent_cache = LatentCache(self.ii2s)
        print(">>> Latent cache loaded")

        self.bald_proxy = BaldProxy(self.g_ema, self.opts.bald_path)
        print(">>> Bald proxy loaded")
        self.text_proxy = TextProxy(self.opts, self.g_ema, self.seg, self.mean_latent_code)
        print(">>> Text proxy loaded")
        self.ref_proxy = RefProxy(self.opts, self.g_ema, self.seg, self.latent_cache)
        print(">>> Ref proxy loaded")
        # self.sketch_proxy = SketchProxy(self.g_ema, self.mean_latent_code, self.opts.sketch_path)
        self.color_proxy = ColorProxy(self.opts, self.g_ema, self.seg)
        print(">>> Color proxy loaded")
        
    def _prepare_src(self, image_path):
        src_latent, src_feature = self.latent_cache.invert_image_in_FS(image_path)
        src_image = Image.open(image_path).convert('RGB')
        src_image = self.image_transform(src_image).unsqueeze(0).cuda()
        return src_latent, src_feature, src_image

    def edit_by_text(self, image_path, text_cond):
        src_latent, src_feature, src_image = self._prepare_src(image_path)
        input_mask = torch.argmax(self.seg(src_image)[1], dim=1).long().clone().detach()
        latent_bald, _ = self.bald_proxy(src_latent)
        latent_global, _ = self.text_proxy(text_cond, src_image, from_mean=True)
        src_feature, edited_hairstyle_img = hairstyle_feature_blending(
            self.g_ema, self.seg, src_latent, src_feature, input_mask, latent_bald, latent_global=latent_global)
        edited_hairstyle_img = process_display_input(edited_hairstyle_img)
        self.latent_cache.cache_latent(edited_hairstyle_img, latent_w=src_latent, latent_fs=src_feature)
        return edited_hairstyle_img
    
    def edit_by_ref(self, image_path, ref_path):
        src_latent, src_feature, src_image = self._prepare_src(image_path)
        input_mask = torch.argmax(self.seg(src_image)[1], dim=1).long().clone().detach()
        latent_bald, _ = self.bald_proxy(src_latent)
        latent_global, _ = self.ref_proxy(ref_path, src_image)
        src_feature, edited_hairstyle_img = hairstyle_feature_blending(
            self.g_ema, self.seg, src_latent, src_feature, input_mask, latent_bald, latent_global=latent_global)
        edited_hairstyle_img = process_display_input(edited_hairstyle_img)
        self.latent_cache.cache_latent(edited_hairstyle_img, latent_w=src_latent, latent_fs=src_feature)
        return edited_hairstyle_img
    
    def edit_color(self, image_path, color_cond):
        src_latent, src_feature, src_image = self._prepare_src(image_path)
        visual_color_list, visual_final_list = self.color_proxy(color_cond, src_image, src_latent, src_feature)
        return visual_final_list[-1]
    
if __name__ == "__main__":
    # Run inference once to download the necessary files
    proxy = InferenceProxy()
    import models.bald_proxy.torch_utils.custom_ops
    models.bald_proxy.torch_utils.custom_ops._init()