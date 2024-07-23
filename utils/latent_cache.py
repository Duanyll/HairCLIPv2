import numpy as np
from PIL import Image
import redis
import hashlib
import os
import io
import torch

REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
CACHE_TIME = int(os.environ.get('CACHE_TIME', 0))

class LatentCache:
    def __init__(self, ii2s):
        self.ii2s = ii2s
        if CACHE_TIME > 0:
            print(f"Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
            self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        else:
            self.redis = None
            
    def _get_image_hash(self, image):
        image_bytes = image.tobytes()
        return hashlib.md5(image_bytes).hexdigest()
        
    def invert_image_in_W(self, image):
        if self.redis is not None:
            image_hash = self._get_image_hash(image)
            cache_key = f"latent_w:{image_hash}"
            latent_w = self.redis.get(cache_key)
            if latent_w is not None:
                buffer = io.BytesIO(latent_w)
                return torch.from_numpy(np.load(buffer)).cuda()
        latent_w = self.ii2s.invert_image_in_W(image=image)
        if self.redis is not None:
            buffer = io.BytesIO()
            np.save(buffer, latent_w.detach().cpu().numpy())
            buffer.seek(0)
            self.redis.set(cache_key, buffer.read(), ex=CACHE_TIME)
        return latent_w
    
    def invert_image_in_FS(self, image):
        latent_w = self.invert_image_in_W(image)
        if self.redis is not None:
            image_hash = self._get_image_hash(image)
            cache_key = f"latent_fs:{image_hash}"
            latent_fs = self.redis.get(cache_key)
            if latent_fs is not None:
                buffer = io.BytesIO(latent_fs)
                return latent_w, torch.from_numpy(np.load(buffer)).cuda()
        latent_w.requires_grad_(True)
        latent_w, latent_fs = self.ii2s.invert_image_in_FS(image=image, latent_W=latent_w)
        if self.redis is not None:
            buffer = io.BytesIO()
            np.save(buffer, latent_fs.detach().cpu().numpy())
            buffer.seek(0)
            self.redis.set(cache_key, buffer.read(), ex=CACHE_TIME)
        return latent_w, latent_fs
    
    def cache_latent(self, image, latent_w=None, latent_fs=None):
        if self.redis is None:
            return
        image_hash = self._get_image_hash(image)
        if latent_w is not None:
            buffer = io.BytesIO()
            np.save(buffer, latent_w.detach().cpu().numpy())
            latent_w_key = f"latent_w:{image_hash}"
            self.redis.set(latent_w_key, buffer.read(), ex=CACHE_TIME)
        if latent_fs is not None:
            buffer = io.BytesIO()
            np.save(buffer, latent_fs.detach().cpu().numpy())
            latent_fs_key = f"latent_fs:{image_hash}"
            self.redis.set(latent_fs_key, buffer.read(), ex=CACHE_TIME)