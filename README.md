# Image Processing - HW1

本專案內容包含：
- 影像讀取與centered 10x10裁切
- 點運算：Log Transform、Gamma Transform、Negative
- 影像重採樣：Nearest Neighbor、Bilinear

---
##  專案結構
├── readimage.py # 影像讀取與中心 10x10 輸出
├── point_ops.py # 點運算
├── resample.py # 重採樣 (Nearest Neighbor / Bilinear)
├── data/ # 輸入影像 (需自行放置)
└── out/ # 輸出結果 (程式自動產生)

## ⚙️ 環境需求
- Python 3.8+
- [NumPy](https://numpy.org/)  
- [Pillow](https://pillow.readthedocs.io/)  


## 使用方式

1. 影像讀取
   
python readimage.py --data data --out-previews out/previews --out-centers out/centers

輸出：
灰階預覽圖
中心 10x10 子區塊 CSV

2. 點運算

python point_ops.py --data data --out out/point_ops

輸出：

{檔名}_log.png
{檔名}_negative.png
{檔名}_gamma_grid.png
{檔名}_overview.png


3. 重採樣

python resample.py --data data --out out/resample

輸出:(包含範例)

512x512 → 128x128
512x512 → 32x32（再放大顯示）
32x32 → 512x512
512x512 → 1024x512
128x128 → 256x512


## 備註
輸入影像 (data/) 與輸出影像 (out/) 因檔案大小未附上，請自行準備測試影像。

.exe 執行檔執行時請確保與輸入影像位於同一資料夾中。

yaml
複製程式碼
