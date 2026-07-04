# 排行榜截图处理工具 - by https://github.com/Shinonome-ena
import customtkinter as ctk
from tkinter import filedialog
import threading
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import ctypes

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

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except:
        pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImageProcessor:
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

    def load_cv(self, path):
        with open(str(path), "rb") as f:
            buf = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def find_cards(self, img_cv):
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
            if row_scores[i] - row_scores[i - 1] <= 5:
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
                cw = right_x - left_x
                ch = bottom_y - top_y
                ratio = cw / ch if ch > 0 else 0
                if 3 < ratio < 15 and cw > 500:
                    cards.append({'x': left_x, 'y': top_y, 'w': cw, 'h': ch})
        return cards

    def create_rounded_mask(self, w, h):
        r = min(h // 4, 20)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
        cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
        cv2.circle(mask, (r, r), r, 255, -1)
        cv2.circle(mask, (w - r, r), r, 255, -1)
        cv2.circle(mask, (r, h - r), r, 255, -1)
        cv2.circle(mask, (w - r, h - r), r, 255, -1)
        return mask

    def detect_tag(self, arr, w, h, start_row=1):
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
            if np.sum(arr[y, :scan_w, 3] > 0) == 0:
                break
            white_count = np.sum(np.all(row > 200, axis=1))
            if white_count >= scan_w * 0.95:
                break
            diff = np.abs(row.astype(int) - ref_color)
            if np.sum(np.all(diff < 50, axis=1)) > 0 or white_count < scan_w * 0.95:
                tag_h = y + 1

        if tag_h < start_row + 2:
            return None
        return {'x': 0, 'y': 0, 'w': scan_w, 'h': tag_h}

    def process_single(self, img_cv, keep_rows=True):
        cards = self.find_cards(img_cv)
        results = []
        extra = 1 if keep_rows else 0

        for card in cards:
            cx, cy, cw, ch = card['x'], card['y'], card['w'], card['h']
            ey = max(0, cy - extra)
            eh = ch + (cy - ey) + extra

            cropped_bgr = img_cv[ey:ey+eh, cx:cx+cw].copy()
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            mask = self.create_rounded_mask(cw, eh)
            rgba = np.zeros((eh, cw, 4), dtype=np.uint8)
            rgba[:, :, :3] = cropped_rgb
            rgba[:, :, 3] = mask

            start_row = 1 if keep_rows else 0
            mirror_offset = 1 if keep_rows else 0
            paste_offset = 1 if keep_rows else 0
            size_extra = 2

            tag = self.detect_tag(rgba, cw, eh, start_row=start_row)
            if tag:
                tx, ty, tw, th = tag['x'], tag['y'], tag['w'], tag['h']
                tw = min(tw + size_extra, cw)
                th = min(th + size_extra, eh)

                src_y = eh - mirror_offset - (th - 1)
                dst_y = ty + paste_offset
                copy_h = th - 1

                if src_y >= 0 and dst_y + copy_h <= eh:
                    patch = rgba[src_y:src_y+copy_h, tx:tx+tw].copy()
                    patch = np.flipud(patch)
                    rgba[dst_y:dst_y+copy_h, tx:tx+tw] = patch

                if keep_rows and tw < cw:
                    fill_color = rgba[0, tw, :]
                    for px in range(tw - 1, -1, -1):
                        if rgba[0, px, 3] == 0:
                            break
                        rgba[0, px] = fill_color

            results.append(_embed(Image.fromarray(rgba)))

        return results

    def process_batch(self, input_files, output_dir, keep_rows, progress_callback=None):
        output_dir.mkdir(exist_ok=True)
        total = len(input_files)
        count = 0

        for idx, img_path in enumerate(input_files, 1):
            try:
                img_cv = self.load_cv(img_path)
                if img_cv is None:
                    continue
                results = self.process_single(img_cv, keep_rows=keep_rows)
                name = Path(img_path).stem
                for i, result in enumerate(results):
                    result.save(output_dir / f"{name}_strip_{i}.png")
                count += 1
            except Exception as e:
                print(f"  错误: {Path(img_path).name} - {e}")
            if progress_callback:
                progress_callback(idx, total)

        return count

    def trim_rows_batch(self, input_files, output_dir, progress_callback=None):
        output_dir.mkdir(exist_ok=True)
        total = len(input_files)
        count = 0

        for idx, f in enumerate(input_files, 1):
            try:
                img = Image.open(f).convert("RGBA")
                arr = np.array(img)
                if img.height > 2:
                    trimmed = arr[1:-1, :, :]
                    _embed(Image.fromarray(trimmed)).save(output_dir / Path(f).name)
                else:
                    img.save(output_dir / Path(f).name)
                count += 1
            except Exception as e:
                print(f"  错误: {Path(f).name} - {e}")
            if progress_callback:
                progress_callback(idx, total)

        return count


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("排行榜截图处理工具 - by https://github.com/Shinonome-ena")
        self.minsize(520, 480)
        self.geometry("520x480")

        self.processor = ImageProcessor()
        self.input_dir = ""
        self.input_files = []
        self.output_dir = ""

        self.create_widgets()

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(6, weight=1)

        title_label = ctk.CTkLabel(self, text="排行榜截图处理工具", font=("", 18, "bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(15, 2), sticky="ew")

        desc_label = ctk.CTkLabel(self, text="自动裁切圆角矩形 · 去除\"你\"标签 · 抠图透明化", font=("", 11), text_color="gray")
        desc_label.grid(row=1, column=0, padx=20, pady=(0, 2), sticky="ew")

        credit_label = ctk.CTkLabel(self, text="by https://github.com/Shinonome-ena", font=("", 9), text_color="gray")
        credit_label.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        dir_frame = ctk.CTkFrame(self)
        dir_frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        dir_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(dir_frame, text="输入目录:").grid(row=0, column=0, padx=(15, 5), pady=(12, 3), sticky="w")
        self.input_entry = ctk.CTkEntry(dir_frame, placeholder_text="选择文件夹或图片文件...")
        self.input_entry.grid(row=0, column=1, padx=5, pady=(12, 3), sticky="ew")
        btn_frame = ctk.CTkFrame(dir_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=2, padx=(5, 15), pady=(12, 3))
        ctk.CTkButton(btn_frame, text="文件夹", width=50, command=self.select_input_folder).pack(side="left", padx=(0, 2))
        ctk.CTkButton(btn_frame, text="文件", width=40, command=self.select_input_file).pack(side="left")

        ctk.CTkLabel(dir_frame, text="输出目录:").grid(row=1, column=0, padx=(15, 5), pady=(3, 12), sticky="w")
        self.output_entry = ctk.CTkEntry(dir_frame, placeholder_text="点击浏览选择文件夹...")
        self.output_entry.grid(row=1, column=1, padx=5, pady=(3, 12), sticky="ew")
        ctk.CTkButton(dir_frame, text="浏览", width=60, command=self.select_output).grid(row=1, column=2, padx=(5, 15), pady=(3, 12))

        mode_frame = ctk.CTkFrame(self)
        mode_frame.grid(row=4, column=0, padx=20, pady=5, sticky="ew")

        ctk.CTkLabel(mode_frame, text="处理模式:", font=("", 12, "bold")).grid(row=0, column=0, padx=15, pady=(12, 5), sticky="w")
        self.mode_var = ctk.StringVar(value="keep")
        ctk.CTkRadioButton(mode_frame, text="留有首末行", variable=self.mode_var, value="keep").grid(row=1, column=0, padx=30, pady=3, sticky="w")
        ctk.CTkRadioButton(mode_frame, text="不留首末行", variable=self.mode_var, value="trim").grid(row=2, column=0, padx=30, pady=3, sticky="w")
        ctk.CTkRadioButton(mode_frame, text="仅首末行裁切", variable=self.mode_var, value="trimonly").grid(row=3, column=0, padx=30, pady=(3, 12), sticky="w")

        self.process_btn = ctk.CTkButton(self, text="开始处理", height=38, font=("", 14, "bold"), command=self.start_process)
        self.process_btn.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.grid(row=6, column=0, padx=20, pady=(0, 15), sticky="sew")
        bottom_frame.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(bottom_frame)
        self.progress.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.progress.set(0)

        status_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(status_frame, text="就绪", font=("", 11), text_color="gray")
        self.status_label.grid(row=0, column=0, sticky="w")

        self.file_count_label = ctk.CTkLabel(status_frame, text="", font=("", 11), text_color="gray")
        self.file_count_label.grid(row=0, column=1, sticky="e")

    def select_input_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if folder:
            self.input_dir = folder
            self.input_files = []
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, folder)
            count = sum(1 for _ in Path(folder).iterdir() if _.suffix.lower() in self.processor.VALID_EXTENSIONS)
            self.file_count_label.configure(text=f"包含 {count} 张图片")

    def select_input_file(self):
        files = filedialog.askopenfilenames(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"), ("所有文件", "*.*")]
        )
        if files:
            self.input_files = list(files)
            self.input_dir = ""
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, f"已选择 {len(files)} 个文件" if len(files) > 1 else files[0])
            self.file_count_label.configure(text=f"已选择 {len(files)} 张图片")

    def select_output(self):
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            self.output_dir = folder
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, folder)

    def get_input_files(self):
        if self.input_files:
            return [Path(f) for f in self.input_files]
        elif self.input_dir:
            p = Path(self.input_dir)
            if not p.exists():
                return []
            return [f for f in p.iterdir() if f.suffix.lower() in self.processor.VALID_EXTENSIONS]
        return []

    def check_output_dir(self, input_files, output_dir):
        output_path = Path(output_dir).resolve()

        if self.input_dir:
            input_path = Path(self.input_dir).resolve()
        elif self.input_files:
            input_path = Path(self.input_files[0]).resolve().parent
        else:
            return output_path, None

        if input_path == output_path:
            sub_dir = output_path / "output"
            if sub_dir.exists():
                from datetime import datetime
                now = datetime.now()
                sub_dir = output_path / f"output({now.month}{now.day}_{now.strftime('%H%M%S')})"
            sub_dir.mkdir(parents=True, exist_ok=True)
            return sub_dir, f"输出目录与输入目录相同，已自动创建子目录: {sub_dir.name}"

        return output_path, None

    def start_process(self):
        input_files = self.get_input_files()
        if not input_files:
            self.status_label.configure(text="请先选择输入目录或文件!")
            return
        if not self.output_dir:
            self.status_label.configure(text="请先选择输出目录!")
            return

        actual_output, warning_msg = self.check_output_dir(input_files, self.output_dir)

        mode = self.mode_var.get()
        self.process_btn.configure(state="disabled", text="处理中...")
        self.progress.set(0)

        def progress_cb(cur, total):
            p = cur / total if total > 0 else 0
            t = f"已处理 {cur}/{total}"
            self.after(0, lambda p=p, t=t: (self.progress.set(p), self.status_label.configure(text=t)))

        def run():
            try:
                if mode == "trimonly":
                    count = self.processor.trim_rows_batch(input_files, actual_output, progress_callback=progress_cb)
                else:
                    keep_rows = mode == "keep"
                    count = self.processor.process_batch(input_files, actual_output, keep_rows, progress_callback=progress_cb)
                msg = f"完成! 共 {count} 张图片已处理"
                if warning_msg:
                    msg += f"\n{warning_msg}"
                self.after(0, lambda: (self.progress.set(1.0), self.status_label.configure(text=msg)))
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(text=f"错误: {str(e)}"))
            finally:
                self.after(0, lambda: self.process_btn.configure(state="normal", text="开始处理"))

        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
