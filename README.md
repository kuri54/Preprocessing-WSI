# Preprocessing WSI
WSIをタイル状に分割し、背景やゴミがある画像を取り除いたものを保存する。
x10、x20、x40相当のタイル画像が保存される。

## Update
### Aug 18, 2023
* `scipy.ndimage.morphology` を使用していたことによるエラーメッセージの解決

### Jun 8, 2023
* 最新版のNumpyにおいて出るエラーの解決

### Sep 28, 2021
* デフォルトでx10のタイル画像を保存するように修正  
それに伴ったコードの微修正

### Sep 27, 2021
* 背景除去とノイズ除去を行うコード（keep_tile）をマルチプロセスで動くようにコードを修正  
    サンプルスライド（tile枚数1900枚）では約25分 -> 約5分へ改善

## 対応フォーマット
* Aperio (.svs, .tif)
* Hamamatsu (.ndpi, .vms, .vmu)
* Leica (.scn)
* MIRAX (.mrxs)
* Philips (.tiff)
* Sakura (.svslide)
* Trestle (.tif)
* Ventana (.bif, .tif)
* Generic tiled TIFF (.tif)

## Rquirements
下記の環境で動作確認済み（全てのライブラリが最新版のもので動く）
* Python 3.7.4
* openslide-python
* numpy
* pandas
* Pillow
* scipy
* scikit-image
* rich
* tqdm

## Installation
1. [Install OpenSlide](https://openslide.org/download/)  
    Linux  
    1-1. `sudo apt-get install build-essential`   
    1-2. `sudo apt-get install openslide-tools`  
    1-3. `sudo apt-get install python-openslide` 

2. `pip install openslide-python`

## フォルダ構成
<pre>
.   
├─ input .. 処理したいWSIを入れるフォルダ  
├─ output .. 処理後のタイル画像が保存されるフォルダ  
├─ notebooks .. コード確認用のnotebooks  
└─ src  
　　└─ preprocessing.py
</pre>
 
## Usage
1. `cd src`
2. `python preprocessing.py`

## Example
* 細胞診などの余白が多い標本を処理する場合  
`python preprocessing.py --tissue_threshold 0.3`

* x20とx40倍相当のタイル画像のみを保存したい場合  
`python preprocessing.py --sample_size 256 512`

## Argument
`python preprocessing.py --help`
