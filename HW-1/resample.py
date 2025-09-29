import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 讀取圖片長寬是否符合
def read_raw_uint8(path: Path, W: int, H: int) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size != W*H: raise ValueError(f"RAW size mismatch: {buf.size} != {W*H}")
    return buf.reshape((H, W))
# 轉灰階存成陣列回傳
def read_standard_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))
# 把陣列轉回影像，儲存結果
def save_png(img_u8: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(path)

# 進行坐標系轉移
def _map_xy(Hs, Ws, Hd, Wd):
    ys = (np.arange(Hd)+0.5)*(Hs/Hd) - 0.5 #長度為原圖H之陣列，為新圖對應舊圖之y座標
    xs = (np.arange(Wd)+0.5)*(Ws/Wd) - 0.5 #長度為原圖W之陣列，為新圖對應舊圖之x座標
    return ys, xs
#nearest neighbor縮放:取出原圖像素，得到新影像(大小為:(new_h,new_w))
def resize_nearest(src: np.ndarray, Wd: int, Hd: int) -> np.ndarray:
    Hs, Ws = src.shape
    ys, xs = _map_xy(Hs, Ws, Hd, Wd)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, Hs-1) #將浮點座標四捨五入成整數並且修至合理範圍
    xi = np.clip(np.rint(xs).astype(np.int32), 0, Ws-1) #將浮點座標四捨五入成整數並且修至合理範圍
    Y, X = np.meshgrid(yi, xi, indexing='ij') #指定(row,col)
    return src[Y, X].astype(np.uint8)
#bilinear縮放
def resize_bilinear(src: np.ndarray, Wd: int, Hd: int) -> np.ndarray:
    Hs, Ws = src.shape
    ys, xs = _map_xy(Hs, Ws, Hd, Wd)
    y0 = np.floor(ys).astype(np.int32); #取每個浮點座標左上角整數格
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0+1, 0, Hs-1); #右下角整數格座標(左下角+1)，並邊界保護
    x1 = np.clip(x0+1, 0, Ws-1)
    wy = (ys - y0).astype(np.float32).reshape(-1,1) #垂直方向距離上邊的距離，越靠近下方越接近1
    wx = (xs - x0).astype(np.float32).reshape(1,-1) #水平方向距離左邊的距離，越靠近右邊越接近1
    Y0, X0 = np.meshgrid(y0, x0, indexing='ij') #(Y0,X0):左上;(Y0,X1):右上;(Y1,X0):左下;(Y1,X1):右下
    Y1, X1 = np.meshgrid(y1, x1, indexing='ij')

    #取出四個鄰近像素值
    Ia = src[Y0, X0].astype(np.float32)#左上
    Ib = src[Y0, X1].astype(np.float32)#右上
    Ic = src[Y1, X0].astype(np.float32)#左下
    Id = src[Y1, X1].astype(np.float32)#右下

    #左右線性插植
    top = Ia*(1-wx)+Ib*wx #上邊線內插
    bot = Ic*(1-wx)+Id*wx #下邊線內插
    out = top*(1-wy)+bot*wy #上下內插
    return np.clip(out+0.5, 0, 255).astype(np.uint8) #+0.5功用類似於為了四捨五入

#確保ascii箭頭能正常顯示，但後來直接把ascii改掉了所以不會用到
def ascii_label(text: str) -> str:
    return (text.replace('×','x').replace('→','->')
                .replace('−','-').replace('–','-').replace('—','-'))

#在結果圖上產生標註
def put_label(img_u8: np.ndarray, text: str) -> np.ndarray:
    h,w = img_u8.shape
    im = Image.fromarray(img_u8).convert("RGBA")
    dr = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("arial.ttf", size=max(18,int(h*0.045)))
    except: font = ImageFont.load_default()
    text = ascii_label(text)
    tw, th = dr.textbbox((0,0), text, font=font)[2:]
    pad = max(6, int(h*0.012))
    dr.rectangle((0,0, min(w, tw+pad*2), th+pad*2), fill=(0,0,0,90))
    dr.text((pad+1,pad+1), text, font=font, fill=(0,0,0,255))
    dr.text((pad,  pad),   text, font=font, fill=(255,255,255,255))
    return np.array(im.convert("L"))

#把影像調整到要的大小後再貼標籤否則標籤字會超大
def label_after_resize(img_u8: np.ndarray, text: str, target_h: int = 512) -> np.ndarray:
    h, w = img_u8.shape
    if h != target_h:
        scale = target_h / h
        new_w = int(round(w * scale))
        img_u8 = resize_nearest(img_u8, new_w, target_h)  
    return put_label(img_u8, text)

#把圖片拼接在一起做比較<若高度不符無法拼接會先將結果圖縮放至相同高度再做拼接
def hcat_same_height(imgs):
    H = max(i.shape[0] for i in imgs)
    pil = []
    for a in imgs:
        if a.shape[0] != H:
            scale = H / a.shape[0]
            W = int(round(a.shape[1]*scale))
            a = resize_nearest(a, W, H)  # self nearest, not PIL
        pil.append(Image.fromarray(a))
    totw = sum(im.width for im in pil)
    out = Image.new("L", (totw, H)); x=0
    for im in pil: out.paste(im,(x,0)); x+=im.width
    return np.array(out)

def main():
    ap = argparse.ArgumentParser() #設定成可變動參數以備不時之需，若沒輸入則自動設定為預設
    ap.add_argument("--data", default="data")
    ap.add_argument("--raw-width", type=int, default=512)
    ap.add_argument("--raw-height", type=int, default=512)
    ap.add_argument("--raw-ext", nargs="+", default=[".raw"])
    ap.add_argument("--out", default="out/resample")
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

            # i) 512x512->128x128
            nn128 = resize_nearest(img,128,128)
            bl128 = resize_bilinear(img,128,128)
            save_png(nn128, out/f"{p.stem}_to128_nn.png")
            save_png(bl128, out/f"{p.stem}_to128_bl.png")
            cmp128 = hcat_same_height([
            label_after_resize(img,              "orig 512",            512),
            label_after_resize(nn128,            "nearest 128",         512),
            label_after_resize(bl128,            "bilinear 128",        512),
            ])
            save_png(cmp128, out/f"{p.stem}_to128_compare.png")

            # ii) 512x512->32x32（顯示放大 x8 否則太小看不清楚)
            nn32 = resize_nearest(img,32,32)
            bl32 = resize_bilinear(img,32,32)
            save_png(nn32, out/f"{p.stem}_to32_nn.png")
            save_png(bl32, out/f"{p.stem}_to32_bl.png")
            nn32_show = resize_nearest(nn32,256,256)
            bl32_show = resize_nearest(bl32,256,256)
            cmp32 = hcat_same_height([
            label_after_resize(img,              "orig 512",            512),
            label_after_resize(nn32_show,        "nearest 32 (x8)",     512),
            label_after_resize(bl32_show,        "bilinear 32 (x8)",    512),
            ])
            save_png(cmp32, out/f"{p.stem}_to32_compare.png")

            # iii) 32x32->512x512（由ii結果回放）
            nn32to512 = resize_nearest(nn32,512,512)
            bl32to512 = resize_bilinear(bl32,512,512)
            save_png(nn32to512, out/f"{p.stem}_32to512_nn.png")
            save_png(bl32to512, out/f"{p.stem}_32to512_bl.png")
            cmp32to512 = hcat_same_height([
            label_after_resize(img,              "orig 512",            512),
            label_after_resize(nn32to512,        "nearest 32->512",     512),
            label_after_resize(bl32to512,        "bilinear 32->512",    512),
            ])
            save_png(cmp32to512, out/f"{p.stem}_32to512_compare.png")

            # iv) 512x512->1024x512（寬放大）
            nn1024x512 = resize_nearest(img,1024,512)
            bl1024x512 = resize_bilinear(img,1024,512)
            save_png(nn1024x512, out/f"{p.stem}_to1024x512_nn.png")
            save_png(bl1024x512, out/f"{p.stem}_to1024x512_bl.png")
            cmp1024x512 = hcat_same_height([
            label_after_resize(img,              "orig 512x512",        512),
            label_after_resize(nn1024x512,       "nearest 1024x512",    512),
            label_after_resize(bl1024x512,       "bilinear 1024x512",   512),
            ])

            save_png(cmp1024x512, out/f"{p.stem}_to1024x512_compare.png")

            # 5) 128x128 -> 256x512
            nn128to256x512 = resize_nearest(nn128,256,512)
            bl128to256x512 = resize_bilinear(bl128,256,512)
            save_png(nn128to256x512, out/f"{p.stem}_128to256x512_nn.png")
            save_png(bl128to256x512, out/f"{p.stem}_128to256x512_bl.png")
            cmp128to256x512 = hcat_same_height([
            label_after_resize(img,              "orig 512x512",        512),
            label_after_resize(nn128to256x512,   "nearest 128->256x512",512),
            label_after_resize(bl128to256x512,   "bilinear 128->256x512",512),
            ])
            save_png(cmp128to256x512, out/f"{p.stem}_128to256x512_compare.png")

            print(f"[OK] {p.name:20s} → resample done")
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")

if __name__ == "__main__":
    main()
