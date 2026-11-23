"""FLUX.2 spatial patches in MOSS's channel order (C, patch_h, patch_w)."""


def patchify(latents):
    if latents.ndim != 4 or latents.shape[-2] % 2 or latents.shape[-1] % 2:
        raise ValueError("MOSS-Image VAE latents must be B,C,H,W with even H and W.")
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify(latents):
    b, c, h, w = latents.shape
    return latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
