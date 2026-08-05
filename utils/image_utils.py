from PIL import Image

def resize_by_short_side(image, target_short=512, resample=Image.BICUBIC):
    w, h = image.size
    if min(w, h) < target_short:
        new_w = (w + 63) // 64 * 64
        new_h = (h + 63) // 64 * 64
    else:
        if w < h:
            new_w = target_short
            new_h = int(h * target_short / w)
        else:
            new_h = target_short
            new_w = int(w * target_short / h)
        new_w = (new_w + 63) // 64 * 64
        new_h = (new_h + 63) // 64 * 64
    return image.resize((new_w, new_h), resample=resample)
