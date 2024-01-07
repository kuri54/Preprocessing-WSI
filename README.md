| [日本語](https://github.com/kuri54/Preprocessing-WSI/blob/main/README-JP.md) |

# Preprocessing WSI
Divide Whole Slide Images (WSI) into tile-like segments and save the images after removing backgrounds and debris.  
Tile images equivalent to x10, x20, and x40 magnifications are stored.

## Update
### Aug 18, 2023
* Solving error messages caused by using `scipy.ndimage.morphology`.

### Jun 8, 2023
* Solving errors that occur with the latest version of Numpy.

### Sep 28, 2021
* Revised to save tile images at x10 magnification by default, along with minor code modifications related to this change.

### Sep 27, 2021
* Revised the code for background and noise removal (keep_tile) to run in a multiprocessing environment.  
For a sample slide (1900 tile images), the processing time improved from approximately 25 minutes to about 5 minutes.

## Supported Formats
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
Confirmed to work in the following environment (compatible with the latest versions of all libraries).
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

## Directory Structure
<pre>
.   
├─ input .. Directory for placing the Whole Slide Images (WSI) to be processed.  
├─ output .. Directory where the tile images are saved after processing.  
├─ notebooks .. Notebooks for code verification 
└─ src  
　　└─ preprocessing.py
</pre>
 
## Usage
1. `cd src`
2. `python preprocessing.py --<arg1> --<arg2>`

## Example
* In cases of processing specimens like cytology samples that have many margins.  
`python preprocessing.py --tissue_threshold 0.3`

* In cases where only tile images equivalent to x20 and x40 magnifications are desired to be saved.  
`python preprocessing.py --sample_size 256 512`

## Argument
`python preprocessing.py --help`
