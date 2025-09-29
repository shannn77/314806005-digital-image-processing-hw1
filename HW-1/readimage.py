import argparse, csv
from pathlib import Path
import numpy as np
from PIL import Image

# 讀取RAW長寬是否符合
def read_raw_uint8(path: Path, W: int, H: int) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size != W*H:
        raise ValueError(f"RAW size mismatch: {buf.size} != {W*H} for {path.name}")
    return buf.reshape((H, W))
#轉灰階存成陣列回傳
def read_standard_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))
#把陣列轉回影像，儲存結果
def save_png(img_u8: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)#避免存檔路徑不存在
    Image.fromarray(img_u8).save(path)

# 進行坐標系轉移
def _map_xy(Hs, Ws, Hd, Wd):
    ys = (np.arange(Hd)+0.5)*(Hs/Hd) - 0.5 #長度為原圖H之陣列，為新圖對應舊圖之y座標
    xs = (np.arange(Wd)+0.5)*(Ws/Wd) - 0.5 #長度為原圖W之陣列，為新圖對應舊圖之x座標
    return ys, xs

#nearest neighbor縮放:取出原圖像素，得到新影像(大小為:(new_h,new_w))
def resize_nearest_np(src_u8: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    Hs, Ws = src_u8.shape
    ys, xs = _map_xy(Hs, Ws, new_h, new_w) #計算新圖像速中心對應在原圖之座標
    #邊界保護
    yi = np.clip(np.rint(ys).astype(np.int32), 0, Hs-1) #將浮點座標四捨五入成整數並且修至合理範圍
    xi = np.clip(np.rint(xs).astype(np.int32), 0, Ws-1) #將浮點座標四捨五入成整數並且修至合理範圍
    Y, X = np.meshgrid(yi, xi, indexing="ij") #指定(row,col)
    return src_u8[Y, X].astype(np.uint8) #取出對應像素轉乘uint8後回傳

#centered 10x10 pixel values
def center_crop_10x10(img: np.ndarray) -> np.ndarray:
    H, W = img.shape
    cy, cx = H//2, W//2 #找到影像的中心
    y0, y1, x0, x1 = cy-5, cy+5, cx-5, cx+5 #由中心向上下左右擴散五個單位得到中心10x10
    if min(y0, x0) < 0 or y1 > H or x1 > W:
        raise ValueError(f"Image too small for 10x10: {W}x{H}") #確保不會超出範圍
    return img[y0:y1, x0:x1] #回傳10x10子陣列

#把10x10陣列存成csv
def save_csv_10x10(arr: np.ndarray, path: Path): #2D陣列
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f: #打開可寫的cvs檔
        w = csv.writer(f)
        for r in arr: #把2D陣列從第一列開始
            w.writerow([int(v) for v in r]) #確保為整數寫成row並進入下一行

def main():
    ap = argparse.ArgumentParser() #設定成可變動參數以備不時之需，若沒輸入則自動設定為預設
    ap.add_argument("--data", default="data")
    ap.add_argument("--raw-width", type=int, default=512)
    ap.add_argument("--raw-height", type=int, default=512)
    ap.add_argument("--raw-ext", nargs="+", default=[".raw"])
    ap.add_argument("--out-previews", default="out/previews")
    ap.add_argument("--out-centers",  default="out/centers")
    args = ap.parse_args()

    #變成路徑
    data = Path(args.data); 
    prev = Path(args.out_previews)  
    ctr  = Path(args.out_centers) 
   
    exts_std = {".bmp", ".jpg", ".jpeg", ".png"}
    exts_raw = set(e.lower() for e in args.raw_ext)

    #列出資料夾下之項目，確保真的有檔案
    files = [p for p in sorted(data.iterdir()) if p.is_file()]
    if not files:
        print("⚠️ data/ is empty.")
        return

    for p in files: #對檔案夾裡的檔案逐檔處理
        ext = p.suffix.lower() #取副檔名，為避免大小寫差異都轉成小寫
        try: #出錯不中斷
            if ext in exts_raw: 
                img = read_raw_uint8(p, args.raw_width, args.raw_height); kind="RAW"   #參數未變動:width與heeight皆等於512
            elif ext in exts_std:
                img = read_standard_gray(p); kind="STD" #標準圖轉灰階
            else:
                continue
            #產出灰階預覽圖片
            save_png(img, prev/f"{p.stem}_preview.png")
            #產出centered 10x10之子陣列
            c10 = center_crop_10x10(img)
            #儲存centered 10x10之子陣列的csv檔案
            save_csv_10x10(c10, ctr/f"{p.stem}_center10.csv")

            #輸出結果
            H, W = img.shape
            print(f"[OK] {p.name:20s} {kind} {W}x{H}")
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")

if __name__ == "__main__":
    main()
