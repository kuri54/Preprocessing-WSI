import os
import math
import argparse
from posixpath import dirname
from tqdm import tqdm
from rich import print

import multiprocessing
from multiprocessing import Pool

import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk

def open_slide(slide_num, input_dir):
    '''画像番号を指定してWSIを開く
    :param slide_num: スライド番号
    :param input_dir: WSIが入っているディレクトリ

    :return OpenSlide objectのWSI
    '''
    slide_names = os.listdir(input_dir)
    filename = os.path.join(input_dir, slide_names[slide_num])

    slide = openslide.open_slide(filename)

    return slide

def create_tile_generator(slide, tile_size, overlap):
    '''WSIからタイル画像を抽出するためのgeneratorを生成する

    :param slide: OpenSlide objectのWSI
    :param tile_size: 生成されるタイル画像の幅と高さを指定（正方形）
    :param overlap: タイル間のオーバーラップのピクセル数

    :return generator
    Note: 抽出されたタイル画像はShape(tile_size, tile_size, channels)を持つ
    '''
    generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)

    return generator

def get_20x_zoom_level(slide, generator):
    '''20倍に相当するzoom levelを返す
    generatorは複数ズームレベルからタイルを抽出
    高解像度から低解像度までの各レベルで2倍のダウンサンプリングをする

    :param slide: OpenSlide objectのWSI
    :param generator: DeepZoomGenerator object

    :return 20倍に相当するズームレベルまたはそれに近いレベル
    '''
    # level_count - 1とすることで最低解像度のレベルindexを取得
    highest_zoom_level = generator.level_count - 1  # 0-based indexing

    try:
        # スライドの対物レンズの倍率を表すプロパティを取得
        mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

        # mag / 20: スライドの倍率と望んだ倍率との間のダウンサンプリング係数
        # (mag / 20) / 2: generatorのダウンサンプリング係数に基づいて、最高解像度レベルからのズームレベルのオフセット
        offset = math.floor((mag / 20) / 2)

        level = highest_zoom_level - offset

    except ValueError:
        # スライドの倍率がわからない場合は最高解像度を返す
        level = highest_zoom_level

    return level

def process_slide(slide_num, input_dir, tile_size, overlap):
    '''WSIに対して可能な全てのタイル画像のインデックスを生成
    を生成

    :param slide_num: スライド番号
    :param input_dir: WSIが入っているディレクトリ
    :param tile_size: 生成されるタイル画像の幅と高さを指定（正方形）
    :param overlap: タイル間のオーバーラップのピクセル数

    :return (slide_num, tile_size, overlap, zoom_level, col, row)のリスト
    '''
    # WSIを開く
    slide = open_slide(slide_num, input_dir)

    # generatorを生成
    generator = create_tile_generator(slide, tile_size, overlap)

    # 20倍のzoom levelを取得
    zoom_level = get_20x_zoom_level(slide, generator)

    # 可能な全てのタイル画像のインデックスを生成
    # 指定したレベルの(tile_x, tile_y)タプルのリストを返す
    cols, rows = generator.level_tiles[zoom_level]

    tile_indices = [(slide_num, tile_size, overlap, zoom_level, col, row)
                    for col in range(cols) for row in range(rows)]

    return tile_indices

def process_tile_index(tile_index, input_dir):
    '''タイルインデックスからタイル画像を生成
    :param tile_index: 抽出するタイルを表すインデックスタプル
    :param input_dir: WSIが入っているディレクトリ

    :return (slide_num, tile)のタプル
    :Note RGB形式の3次元Numpy配列 (tile_size, tile_size, channels)
    '''
    slide_num, tile_size, overlap, zoom_level, col, row = tile_index

    # WSIを開く
    slide = open_slide(slide_num, input_dir)

    # generatorを生成
    generator = create_tile_generator(slide, tile_size, overlap)

    # タイル画像を生成
    tile = np.array(generator.get_tile(zoom_level, (col, row)))

    return (slide_num, tile)

def keep_tile(tile_tuple, tile_size, tissue_threshold):
    '''タイル画像を残すかどうか判断する
    サイズと組織割合の閾値に基づいてタイル画像をフィルタリングする
    あるタイル画像の高さと幅が(tile_size, tile_size)に等しく、
    かつ、指定された％以上の成分を含むタイルを保持する
    それ以外の場合はフィルタリングされる

    :param tile_tuple: (slide_num, tile) のタプル
    :param tile_size: 生成されるタイル画像の幅と高さを指定（正方形）
    :param tissue_threshold: 組織割合の閾値

    :return タイル画像を保持するかどうかのブール値
    '''
    slide_num, tile = tile_tuple

    if tile.shape[0:2] == (tile_size, tile_size):
        # 3DのRGB画像を2Dのgrayscaleに変換
        # 0 (高密度) から1 (無地の背景)
        tile = rgb2gray(tile)

        # 8-bitのDepthを補完 1から0まで
        tile = 1 - tile

        # hysteresis thresholdingによるCanny edgeの検出
        # 1がedgeに相当（組織にはedgeが多い、背景はそうでもないということ）
        tile = canny(tile)

        # Binary closing
        # 拡張の後に収縮を行うことで背景のノイズを取り除く
        tile = binary_closing(tile, disk(10))

        # Binary dilation
        # 明るい部分を拡大し、暗い部分を縮小することで組織領域内の穴を埋める
        tile = binary_dilation(tile, disk(10))

        # 組織領域内の残りの穴を埋める
        tile = binary_fill_holes(tile)

        # 組織のカバー率を算出
        percentage = tile.mean()

        return percentage >= tissue_threshold

    else:
      return False

def process_tile(tile_tuple, sample_size):
    '''タイル画像をより小さな画像に加工する
    タイル画像をsample_size * sample_sizeピクセルの小さなタイルに切り分ける
    各サンプルの形状を(H, W, C)から(C, H, W)に変換する
    それぞれを長さC * H * Wの長さのベクトルに変換

    :param tile_tuple: (slide_num, tile) のタプル
    :param sample_size: 生成されるタイル画像の幅と高さを指定（正方形）

    :return Flattenされた(channels * sample_size_x * sample_size_y)のベクトル
    '''
    slide_num, tile = tile_tuple

    x, y, ch = tile.shape

    # (num_x, sample_size_x, num_y, sample_size_y, ch）の5次元配列にreshape
    # num_x, yはそれぞれx軸とy軸に分割されたタイルの数
    samples = (tile.reshape((x // sample_size, sample_size, y // sample_size, sample_size, ch))
                    .swapaxes(1,2) # sample_size_xとnum_yの軸を入れ替え (num_x, num_y, sample_size_x, sample_size_y, ch)
                    .reshape((-1, sample_size, sample_size, ch)) # num_xとnum_yを1つの軸に結合 (num_samples, sample_size_x, sample_size_y, ch)
                    .transpose(0,3,1,2)) # (num_samples, ch, sample_size_x, sample_size_y)

    # Flatten (num_samples, ch * sample_size_x * sample_size_y)
    samples = samples.reshape(samples.shape[0], -1)

    samples = [(slide_num, sample) for sample in list(samples)]

    return samples

def get_file_name(slide_num, input_dir):
    '''ファイル名を取得
    :return 拡張子を除いたファイル名
    '''
    slide_names = os.listdir(input_dir)
    filename = slide_names[slide_num].split('.')

    # 拡張子はいらない
    return filename[0]

def make_dirs(output_dir, slide_name, sample_size):
    '''保存用のディレクトリを作成し、ディレクトリ名を返す
    :return 保存ディレクトリ名, 倍率名
    '''
    if sample_size == 256:
        dir_name = 'x40'

    elif sample_size == 512:
        dir_name = 'x20'

    elif sample_size == 1024:
        dir_name = 'x10'

    output_dir_name = output_dir + dir_name + '/' + str(slide_name)
    os.makedirs(output_dir_name, exist_ok=True)

    return output_dir_name, dir_name

def save_jpeg_images(sample, sample_num, sample_size, output_dir_name, slide_name):
    '''flattenされたvectorから画像をjpegで保存
    :param sample: (channels * sample_size_x * sample_size_y)
    :param sample_num: サンプルの番号 連番をファイル名につけるために必要
    :param sample_size: 生成されるタイル画像の幅と高さを指定（正方形）
    :param output_dir_name: 保存ディレクトリ名
    :param slide_name: ファイル名
    '''
    # 画像長を抽出し、チャネル数を算出
    length = sample.shape[0]
    channels = int(length / (sample_size * sample_size))

    # 配列を(H*W*C)の形式に変換
    image = sample.astype('uint8').reshape((channels, sample_size, sample_size)).transpose(1,2,0)

    # PILフォーマットに変換
    pil_image = Image.fromarray(image)

    # 保存 WSI名_連番の形式で
    pil_image.save(f'{output_dir_name}/{slide_name}_{sample_num}.jpg')

def wrapper_keep_tile(args):
    '''keep_tileのラッパー
    multiprocessing時に必要
    '''
    return keep_tile(*args)

def pre_process(slide_num, input_dir, tile_size, over_lap, tissue_threshold):
    '''
    :param slide_num: スライド番号
    :param input_dir: WSIが入っているディレクトリ
    :param tile_size: 生成されるタイル画像の幅と高さを指定（正方形）
    :param over_lap: タイル間のオーバーラップのピクセル数
    :param tissue_threshold: 組織割合の閾値

    :return フィルタリングされたタイル画像
    '''
    # ファイル名を取得
    slide_name = get_file_name(slide_num, input_dir)

    # WSIに対して可能な全てのタイル画像のインデックスを生成
    print(f'Process start....: [u]{slide_name}[/u]')
    tile_idx = process_slide(slide_num, input_dir, tile_size, over_lap)

    # タイルインデックスからタイル画像を生成
    print('[bold green]Generate tiled image from index....')
    tiles = [process_tile_index(i, input_dir) for i in tqdm(tile_idx)]

    # タイル画像のフィルタリング -> multiprocessingで速く処理
    # 引数をいじれるようにargsにまとめて処理させる
    print('[bold green]Filtering a tile image....')
    args = [[i, tile_size, tissue_threshold] for i in tiles]

    with Pool(multiprocessing.cpu_count()) as pool:
        filtered_tiles = [i for i, keep in zip(tiles, pool.map(wrapper_keep_tile, tqdm(args))) if keep]

    return filtered_tiles

def post_process(filtered_tiles, slide_num, input_dir, output_dir, sample_size):
    '''
    :param filtered_tiles: フィルタリングされたタイル画像
    :param slide_num: スライド番号
    :param input_dir: WSIが入っているディレクトリ
    :param output_dir: タイル画像を保存するディレクトリ
    :param sample_size: 生成されるタイル画像の幅と高さを指定（正方形）
    '''
    # ファイル名を取得
    slide_name = get_file_name(slide_num, input_dir)

    # タイル画像をより小さなタイル画像にする
    samples = [n for i in filtered_tiles for n in process_tile(i, sample_size)]

    # 保存用のディレクトリを作成してディレクトリ名と倍率名を取得
    output_dir_name, dir_name = make_dirs(output_dir, slide_name, sample_size)

    # flatten vectorから画像をPILフォーマットに変換し保存
    print(f'[bold red]Saving {dir_name}zoom Level Tile Images....')
    for sample_num, sample in enumerate(tqdm(samples)):
        slide_num, sample = sample
        save_jpeg_images(sample, sample_num, sample_size, output_dir_name, slide_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', default='../input/', help='pathの最後に/をつけることを忘れずに')
    parser.add_argument('-o', '--output_dir', default='../output/', help='pathの最後に/をつけることを忘れずに')
    parser.add_argument('-t', '--tile_size', default=1024, type=int)
    parser.add_argument('-l', '--over_lap', default=0, type=int)
    parser.add_argument('-u', '--tissue_threshold', default=0.9, type=float, help='数値を小さくするほど余白部分を含んだ画像を残すようになる')
    parser.add_argument('-s', '--sample_size', nargs='*', default=[256, 512, 1024], help='固定倍率のみを保存したい場合 -> x40:256, x20:512, x10:1024を引数指定する')

    args = parser.parse_args()

    for slide_num, _ in enumerate(os.listdir(args.input_dir)):
        filtered_tiles = pre_process(
            slide_num=slide_num,
            input_dir=args.input_dir,
            tile_size=args.tile_size,
            over_lap=args.over_lap,
            tissue_threshold=args.tissue_threshold
            )

        for arg in args.sample_size:
            post_process(
                filtered_tiles,
                slide_num=slide_num,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                sample_size=int(arg) # int型にしておかないとmake_dirsでエラーが出る
                )