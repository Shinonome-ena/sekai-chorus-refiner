"""验证脚本 — 检测图片中是否含有LSB隐写溯源信息"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image

_AUTHOR_MSG = "Shinonome-ena | https://github.com/Shinonome-ena"

def _bits_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def _extract(pil_img):
    arr = np.array(pil_img)
    flat_rgb = arr[:, :, :3].flatten()
    if len(flat_rgb) < 32:
        return None
    header_bits = ''.join(str(flat_rgb[i] & 1) for i in range(32))
    msg_len = int(header_bits, 2)
    if msg_len <= 0 or msg_len > len(flat_rgb) - 32:
        return None
    msg_bits = ''.join(str(flat_rgb[32 + i] & 1) for i in range(msg_len))
    result = _bits_to_str(msg_bits)
    return result if result else None


def verify(path):
    p = Path(path)
    if not p.exists():
        print(f"  文件不存在: {path}")
        return False

    img = Image.open(p).convert("RGBA")
    msg = _extract(img)
    if msg:
        print(f"  [FOUND] {p.name}")
        print(f"  内容: {msg}")
        if msg == _AUTHOR_MSG:
            print(f"  来源: Shinonome-ena 的排行榜截图处理工具")
        return True
    else:
        print(f"  [NONE] {p.name} — 未检测到隐写信息")
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_stego.py <图片路径或文件夹>")
        print("示例: python verify_stego.py output/")
        print("      python verify_stego.py image.png")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        exts = {'.png', '.jpg', '.jpeg'}
        files = sorted(f for f in target.iterdir() if f.suffix.lower() in exts)
        if not files:
            print(f"  文件夹内无图片: {target}")
            sys.exit(1)
        found = 0
        for f in files:
            if verify(f):
                found += 1
        print(f"\n  结果: {found}/{len(files)} 张图片含有溯源信息")
    else:
        verify(target)


if __name__ == "__main__":
    main()
