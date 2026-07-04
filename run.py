import sys
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import shutil

# ── LSB 隐写 ──
_AUTHOR_MSG = "Shinonome-ena | https://github.com/Shinonome-ena"

def _str_to_bits(s):
    return ''.join(format(b, '08b') for b in s.encode('utf-8'))

def _bits_to_str(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def _embed(pil_img, msg=_AUTHOR_MSG):
    arr = np.array(pil_img).copy()
    h, w = arr.shape[:2]
    bits = _str_to_bits(msg)
    header = format(len(bits), '032b')
    all_bits = header + bits
    if len(all_bits) > h * w * 3:
        raise ValueError(f"消息太长: {len(all_bits)} bits > 可用 {h*w*3} bits")
    flat_rgb = arr[:, :, :3].flatten()
    for i, bit in enumerate(all_bits):
        flat_rgb[i] = (flat_rgb[i] & 0xFE) | int(bit)
    arr[:, :, :3] = flat_rgb.reshape(h, w, 3)
    return Image.fromarray(arr)

def select_folder(title="选择文件夹"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder if folder else None

def load_cv(path):
    with open(str(path), "rb") as f:
        buf = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def find_card_by_structure(img_cv):
    h, w = img_cv.shape[:2]
    white_mask = np.all(img_cv > 240, axis=2)
    gray_mask = (img_cv[:, :, 0] > 220) & (img_cv[:, :, 1] > 220) & (img_cv[:, :, 2] > 220) & (~white_mask)
    row_scores = []
    for y in range(h):
        white_xs = np.where(white_mask[y])[0]
        gray_xs = np.where(gray_mask[y])[0]
        if len(white_xs) > 0 and len(gray_xs) > 0:
            if np.mean(white_xs) < np.mean(gray_xs) and len(white_xs) > w * 0.02 and len(gray_xs) > w * 0.02:
                row_scores.append(y)
    if not row_scores:
        return []
    groups = []
    current_group = [row_scores[0]]
    for i in range(1, len(row_scores)):
        if row_scores[i] - row_scores[i-1] <= 5:
            current_group.append(row_scores[i])
        else:
            if len(current_group) > 50:
                groups.append((current_group[0], current_group[-1]))
            current_group = [row_scores[i]]
    if len(current_group) > 50:
        groups.append((current_group[0], current_group[-1]))
    cards = []
    for top_y, bottom_y in groups:
        mid_y = (top_y + bottom_y) // 2
        white_xs = np.where(white_mask[mid_y])[0]
        gray_xs = np.where(gray_mask[mid_y])[0]
        if len(white_xs) > 0 and len(gray_xs) > 0:
            left_x = white_xs[0]
            right_x = gray_xs[-1]
            card_w = right_x - left_x
            card_h = bottom_y - top_y
            ratio = card_w / card_h if card_h > 0 else 0
            if 3 < ratio < 15 and card_w > 500:
                cards.append({'x': left_x, 'y': top_y, 'w': card_w, 'h': card_h})
    return cards

def create_rounded_mask(w, h):
    corner_r = min(h // 4, 20)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (corner_r, 0), (w - corner_r, h), 255, -1)
    cv2.rectangle(mask, (0, corner_r), (w, h - corner_r), 255, -1)
    cv2.circle(mask, (corner_r, corner_r), corner_r, 255, -1)
    cv2.circle(mask, (w - corner_r, corner_r), corner_r, 255, -1)
    cv2.circle(mask, (corner_r, h - corner_r), corner_r, 255, -1)
    cv2.circle(mask, (w - corner_r, h - corner_r), corner_r, 255, -1)
    return mask

def step1_crop(input_dir, output_dir, keep_rows=True):
    output_dir.mkdir(exist_ok=True)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(ext))
    files.sort(key=lambda f: f.name.lower())
    total = len(files)

    extra = 1 if keep_rows else 0

    print(f"\n  步骤1: 裁切圆角矩形 (extra={extra})")
    print(f"  输入: {input_dir.name} ({total} 张)")
    print(f"  输出: {output_dir.name}")

    success = 0
    fail = 0

    for idx, img_file in enumerate(files, 1):
        try:
            img_cv = load_cv(img_file)
            if img_cv is None:
                fail += 1
                continue

            cards = find_card_by_structure(img_cv)

            for i, card in enumerate(cards):
                cx, cy, cw, ch = card['x'], card['y'], card['w'], card['h']
                ey = max(0, cy - extra)
                eh = ch + (cy - ey) + extra
                cropped = img_cv[ey:ey+eh, cx:cx+cw].copy()

                rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                mask = create_rounded_mask(cw, eh)
                rgba = np.zeros((eh, cw, 4), dtype=np.uint8)
                rgba[:, :, :3] = rgb
                rgba[:, :, 3] = mask

                result = Image.fromarray(rgba)
                result = _embed(result)
                name = Path(img_file).stem
                result.save(output_dir / f"{name}_strip_{i}.png")

            success += 1
            if idx % 20 == 0 or idx == total:
                print(f"  进度: {idx}/{total}")
        except Exception as e:
            fail += 1

    print(f"  完成: {success} 成功, {fail} 失败")
    return success

def detect_tag_boundary(arr, w, h, start_row=1):
    sample_rows = min(5, h)
    ref_x = None
    ref_color = None
    start_x = math.ceil(w * 0.2)

    for sy in range(start_row, sample_rows):
        r, g, b, a = arr[sy, start_x]
        if a > 0 and not (r > 200 and g > 200 and b > 200):
            start_x = math.ceil(w / 4)

        for sx in range(start_x, -1, -1):
            r, g, b, a = arr[sy, sx]
            if a > 0 and not (r > 200 and g > 200 and b > 200):
                ref_x = sx
                ref_color = np.array([r, g, b], dtype=int)
                break
        if ref_x is not None:
            break

    if ref_x is None:
        return None

    scan_w = ref_x + 1
    tag_h = 0

    for y in range(start_row, h):
        row = arr[y, :scan_w, :3]
        visible = np.sum(arr[y, :scan_w, 3] > 0)
        if visible == 0:
            break
        white_count = np.sum(np.all(row > 200, axis=1))
        if white_count >= scan_w * 0.95:
            break
        diff = np.abs(row.astype(int) - ref_color)
        match_count = np.sum(np.all(diff < 50, axis=1))
        if match_count > 0 or white_count < scan_w * 0.95:
            tag_h = y + 1

    if tag_h < start_row + 2:
        return None
    return {'x': 0, 'y': 0, 'w': scan_w, 'h': tag_h}

def step2_remove_tag(input_dir, output_dir, keep_rows=True):
    output_dir.mkdir(exist_ok=True)
    files = sorted(input_dir.glob("*.png"))
    total = len(files)

    start_row = 1 if keep_rows else 0
    mirror_offset = 1 if keep_rows else 0
    paste_offset = 1 if keep_rows else 0
    size_extra = 2

    print(f"\n  步骤2: 去除'你'标签 (start_row={start_row})")
    print(f"  输入: {input_dir.name} ({total} 张)")
    print(f"  输出: {output_dir.name}")

    success = 0
    skip = 0

    for idx, f in enumerate(files, 1):
        try:
            img = Image.open(f).convert("RGBA")
            arr = np.array(img)
            w, h = img.size

            tag = detect_tag_boundary(arr, w, h, start_row=start_row)

            if tag:
                tx, ty, tw, th = tag['x'], tag['y'], tag['w'], tag['h']
                tw = min(tw + size_extra, w)
                th = min(th + size_extra, h)

                src_y = h - mirror_offset - (th - 1)
                dst_y = ty + paste_offset
                copy_h = th - 1

                if src_y >= 0 and dst_y + copy_h <= h:
                    patch = arr[src_y:src_y+copy_h, tx:tx+tw].copy()
                    patch = np.flipud(patch)
                    arr[dst_y:dst_y+copy_h, tx:tx+tw] = patch

                if keep_rows and tw < w:
                    fill_color = arr[0, tw, :]
                    for px in range(tw - 1, -1, -1):
                        if arr[0, px, 3] == 0:
                            break
                        arr[0, px] = fill_color

                success += 1
            else:
                skip += 1

            result = Image.fromarray(arr)
            result = _embed(result)
            result.save(output_dir / f.name)

            if idx % 20 == 0 or idx == total:
                print(f"  进度: {idx}/{total}")
        except Exception as e:
            print(f"  错误 {f.name}: {e}")

    print(f"  完成: {success} 去除标签, {skip} 无标签")
    return success

def step3_trim_rows(input_dir, output_dir):
    output_dir.mkdir(exist_ok=True)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(ext))
    files.sort(key=lambda f: f.name.lower())
    total = len(files)

    print(f"\n  步骤3: 裁去首末行")
    print(f"  输入: {input_dir.name} ({total} 张)")
    print(f"  输出: {output_dir.name}")

    success = 0
    for idx, f in enumerate(files, 1):
        try:
            img = Image.open(f).convert("RGBA")
            arr = np.array(img)
            if img.height > 2:
                trimmed = arr[1:-1, :, :]
                out_img = _embed(Image.fromarray(trimmed))
                out_img.save(output_dir / f.name)
            else:
                img.save(output_dir / f.name)
            success += 1
            if idx % 20 == 0 or idx == total:
                print(f"  进度: {idx}/{total}")
        except Exception as e:
            print(f"  错误 {f.name}: {e}")

    print(f"  完成: {success} 张")
    return success

def show_menu():
    print(f"\n{'='*50}")
    print(f"  排行榜截图处理工具")
    print(f"{'='*50}")
    print(f"  [1] 选择图片目录")
    print(f"  [2] 一键全流程 (留有首末行)")
    print(f"  [3] 一键全流程 (不留首末行)")
    print(f"  [4] 仅裁切圆角矩形")
    print(f"  [5] 仅去除'你'标签")
    print(f"  [6] 仅首末行裁切")
    print(f"  [0] 退出")
    print(f"{'='*50}")

def run_pipeline(input_dir, output_dir, keep_rows):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    temp1 = output_path / "_temp_crop"
    temp2 = output_path / "_temp_tag"

    print(f"\n{'='*50}")
    print(f"  处理: {input_path.name}")
    print(f"  输出: {output_path.name}")
    print(f"  模式: {'留有首末行' if keep_rows else '不留首末行'}")
    print(f"{'='*50}")

    output_path.mkdir(exist_ok=True)
    step1_crop(input_path, temp1, keep_rows=keep_rows)
    step2_remove_tag(temp1, temp2, keep_rows=keep_rows)

    if temp2.exists():
        for f in temp2.glob("*.png"):
            shutil.copy2(f, output_path / f.name)
    count = len(list(output_path.glob("*.png")))

    shutil.rmtree(temp1, ignore_errors=True)
    shutil.rmtree(temp2, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"  完成! 共 {count} 个文件")
    print(f"  输出: {output_path}")
    print(f"{'='*50}")

def main():
    selected_dir = None

    while True:
        show_menu()
        choice = input("\n  请选择操作: ").strip()

        if choice == "0":
            print("\n  再见!")
            break

        elif choice == "1":
            print("\n  请选择图片文件夹...")
            folder = select_folder("选择图片文件夹")
            if folder:
                selected_dir = folder
                file_count = sum(1 for _ in Path(folder).glob("*") if _.suffix.lower() in ['.jpg', '.jpeg', '.png'])
                print(f"  已选择: {folder}")
                print(f"  包含图片: {file_count} 张")
            else:
                print("  已取消")

        elif choice == "2":
            if not selected_dir:
                print("\n  请先选择图片目录 [1]")
                continue
            print("\n  请选择输出文件夹...")
            output_dir = select_folder("选择输出文件夹")
            if not output_dir:
                print("  已取消")
                continue
            run_pipeline(selected_dir, output_dir, keep_rows=True)

        elif choice == "3":
            if not selected_dir:
                print("\n  请先选择图片目录 [1]")
                continue
            print("\n  请选择输出文件夹...")
            output_dir = select_folder("选择输出文件夹")
            if not output_dir:
                print("  已取消")
                continue
            run_pipeline(selected_dir, output_dir, keep_rows=False)

        elif choice == "4":
            if not selected_dir:
                print("\n  请先选择图片目录 [1]")
                continue
            print("\n  请选择输出文件夹...")
            output_dir = select_folder("选择输出文件夹")
            if not output_dir:
                print("  已取消")
                continue
            step1_crop(Path(selected_dir), Path(output_dir))

        elif choice == "5":
            print("\n  请选择裁切后的图片文件夹...")
            input_dir = select_folder("选择输入文件夹")
            if not input_dir:
                print("  已取消")
                continue
            print("\n  请选择输出文件夹...")
            output_dir = select_folder("选择输出文件夹")
            if not output_dir:
                print("  已取消")
                continue
            step2_remove_tag(Path(input_dir), Path(output_dir))

        elif choice == "6":
            print("\n  请选择图片文件夹...")
            input_dir = select_folder("选择输入文件夹")
            if not input_dir:
                print("  已取消")
                continue
            print("\n  请选择输出文件夹...")
            output_dir = select_folder("选择输出文件夹")
            if not output_dir:
                print("  已取消")
                continue
            step3_trim_rows(Path(input_dir), Path(output_dir))

        else:
            print("  无效选择，请重试")

if __name__ == "__main__":
    main()
