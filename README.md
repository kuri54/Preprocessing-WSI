# Preprocessing WSI
WSIをタイル状に分割し、背景やゴミがある画像を取り除いたものを保存する。
x20相当とx40相当のタイル画像が保存される。

## Update
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
下記の環境で動作確認済み（ほぼ全てのライブラリが最新版のもので動く）
* Python 3.7.4
* openslide-python==1.1.2
* numpy==1.17.2
* pandas==0.25.1
* Pillow==6.2.0
* scipy==1.4.1
* scikit-image==0.18.3
* rich==10.10.0
* tqdm==4.36.1

## Installation
1. [Install OpenSlide](https://openslide.org/download/)  
    Linux  
    1-1. `sudo apt-get install build-essential`   
    1-2. `sudo apt-get install openslide-tools`  
    1-3. `sudo apt-get install python-openslide` 

2. `pip install openslide-python`

## フォルダ構成
.   
├─ input .. 処理したいWSIを入れるフォルダ（作成する）  
├─ output .. 処理後のタイル画像が保存されるフォルダ（作成する）  
├─ notebooks .. コード確認用のnotebooks  
└─ src  
　　└─ preprocessing.py

## Usage
1. `cd src`
2. `python preprocessing.py`

## Argument
`python preprocessing.py --help`