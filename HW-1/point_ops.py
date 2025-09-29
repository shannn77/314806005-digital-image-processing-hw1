import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 讀取RAW長寬是否符合
def read_raw_uint8(path: Path, W: int, H: int) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size != W*H: raise ValueError(f"RAW size mismatch: {buf.size} != {W*H}")
    return buf.reshape((H, W))
#轉灰階存成陣列回傳
def read_standard_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))
#把陣列轉回影像，儲存結果
def save_png(img_u8: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(path)

#得到浮點影像避免溢位
def to01(u8): 
    return u8.astype(np.float32)/255.0
#把浮點影像轉回uint8
def to_u8(f):
    return (np.clip(f,0,1)*255.0 + 0.5).astype(np.uint8) #放大回0~255並四捨五入
#log-transform
def log_transform(img):
    f = to01(img)  
    g = np.log1p(f)/np.log(2.0) #對數轉換
    return to_u8(g)
#gamma-transform (gamma<1:變亮; gamma>1:變暗)
def gamma_transform(img, gamma):
    f = to01(img) 
    g = np.power(np.clip(f,0,1), gamma) #將 f clip至[0,1]，再以f為底數gamma為指數得到新像素值
    return to_u8(g)
#image negative
def negative(img): 
    return 255 - img #得到完全相反的pixel value

# 進行坐標系轉移
def _map_xy(Hs, Ws, Hd, Wd):
    ys = (np.arange(Hd)+0.5)*(Hs/Hd) - 0.5 #長度為原圖H之陣列，為新圖對應舊圖之y座標
    xs = (np.arange(Wd)+0.5)*(Ws/Wd) - 0.5 #長度為原圖W之陣列，為新圖對應舊圖之x座標
    return ys, xs

#nearest neighbor縮放:取出原圖像素，得到新影像(大小為:(new_h,new_w))
def resize_nearest_np(src_u8: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    Hs, Ws = src_u8.shape
    ys, xs = _map_xy(Hs, Ws, new_h, new_w)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, Hs-1)
    xi = np.clip(np.rint(xs).astype(np.int32), 0, Ws-1)
    Y, X = np.meshgrid(yi, xi, indexing="ij")
    return src_u8[Y, X].astype(np.uint8)

#確保ascii箭頭能正常顯示，但後來直接把ascii改掉了所以不會用到
def ascii_label(text: str) -> str:
    return (text.replace('×','x').replace('→','->')
                .replace('−','-').replace('–','-').replace('—','-'))
#在結果圖上產生標註
def put_label(img_u8: np.ndarray, text: str) -> np.ndarray:
    h, w = img_u8.shape
    im = Image.fromarray(img_u8).convert("RGBA")
    dr = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("arial.ttf", size=max(18, int(h*0.045)))
    except: font = ImageFont.load_default()
    text = ascii_label(text)
    tw, th = dr.textbbox((0,0), text, font=font)[2:]
    pad = max(6, int(h*0.012))
    dr.rectangle((0,0, min(w, tw+pad*2), th+pad*2), fill=(0,0,0,90))
    dr.text((pad+1,pad+1), text, font=font, fill=(0,0,0,255))
    dr.text((pad,  pad),   text, font=font, fill=(255,255,255,255))
    return np.array(im.convert("L"))

#把圖片橫向拼接在一起做比較，若高度不符無法拼接會先將結果圖縮放至相同高度再做拼接
def hcat_same_height(imgs):
    H = max(i.shape[0] for i in imgs)
    pil = []
    for a in imgs:
        if a.shape[0] != H:
            scale = H / a.shape[0]
            W = int(round(a.shape[1]*scale))
            a = resize_nearest_np(a, W, H)  # self nearest
        pil.append(Image.fromarray(a))
    totw = sum(im.width for im in pil)
    out = Image.new("L", (totw, H)); x=0
    for im in pil: out.paste(im,(x,0)); x+=im.width
    return np.array(out)
#把圖片直向堆疊在一起做比較，若寬度不符無法拼接會先將較窄之右側補色讓寬度達成一致
def vstack_pad_same_width(imgs, fill=0):
    pil_imgs = [Image.fromarray(a) for a in imgs]
    max_w = max(im.width for im in pil_imgs)
    total_h = sum(im.height for im in pil_imgs)
    out = Image.new("L", (max_w, total_h), color=fill)
    y=0
    for im in pil_imgs:
        out.paste(im, (0, y))  # 向左對齊
        y += im.height
    return np.array(out)

def main():
    ap = argparse.ArgumentParser() #設定成可變動參數以備不時之需，若沒輸入則自動設定為預設
    ap.add_argument("--data", default="data")
    ap.add_argument("--raw-width", type=int, default=512)
    ap.add_argument("--raw-height", type=int, default=512)
    ap.add_argument("--raw-ext", nargs="+", default=[".raw"])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.5,0.8,1.2,2.2]) #gamma值預設為[0.5,0.8,1.2,2.2]
    ap.add_argument("--out", default="out/point_ops")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exts_std = {".bmp",".jpg",".jpeg",".png"}
    exts_raw=set(e.lower() for e in args.raw_ext)

    #列出資料夾下之項目，確保真的有檔案
    files = [p for p in sorted(data.iterdir()) if p.is_file()]
    if not files:
        print("⚠️ data/ is empty."); return

    for p in files: #對檔案夾裡的檔案逐檔處理
        ext = p.suffix.lower() #取副檔名，為避免大小寫差異都轉成小寫
        try:
            if ext in exts_raw: img = read_raw_uint8(p,args.raw_width,args.raw_height)
            elif ext in exts_std: img = read_standard_gray(p)
            else: continue

            #執行三種不同的transform
            img_log = log_transform(img)
            gamma_row = hcat_same_height([put_label(gamma_transform(img,g), f"gamma {g}") for g in args.gammas])
            img_neg = negative(img)

            #儲存結果
            save_png(img_log, out/f"{p.stem}_log.png")
            save_png(img_neg, out/f"{p.stem}_negative.png")
            save_png(gamma_row, out/f"{p.stem}_gamma_grid.png")
            #先拼接原圖 log negative    
            top = hcat_same_height([put_label(img,"original"),
                                    put_label(img_log,"log"),
                                    put_label(img_neg,"negative")])
            #再拼接gamma的四張結果圖
            overview = vstack_pad_same_width([top, gamma_row], fill=0)
            save_png(overview, out/f"{p.stem}_overview.png")
            print(f"[OK] {p.name:20s} → outputs done")
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")

if __name__ == "__main__":
    main()
