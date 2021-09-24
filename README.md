# Preprocessing WSI
WSIをタイル状に分割し、背景やゴミがある画像を取り除いたものを保存する。
x20相当とx40相当のタイル画像が保存される。

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

## 引数
`python preprocessing.py --help`