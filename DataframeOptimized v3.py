import numpy as np
import os
import pandas as pd
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from time import time
from pprint import pprint
from math import ceil

# Temporary Workaround
# https://github.com/gtaylor/python-colormath/issues/104#issuecomment-1247481638
if np.__version__ > "1.16":
    def patch_asscalar(a):
        return a.item()
setattr(np, "asscalar", patch_asscalar)


def setRGB(RGB):
    pR, pG, pB = RGB
    sRGB = sRGBColor(pR, pG, pB, is_upscaled=True)
    return sRGB


def convertLAB(sRGB):
    LAB = convert_color(sRGB, LabColor)
    return LAB


def deltaE(testColorLAB, paletteLAB):
    deltaE = abs(delta_e_cie2000(testColorLAB, paletteLAB))
    return deltaE


def newColor(pixelLAB, palette):
    paletteCopy = palette[['Name', 'RGB', 'LAB']].copy()
    paletteCopy['deltaE'] = paletteCopy['LAB'].map(lambda pLAB: deltaE(pixelLAB, pLAB))
    minDeltaEIDX = paletteCopy['deltaE'].idxmin(axis = 0)
    closestPalette = paletteCopy.loc[paletteCopy.index[minDeltaEIDX], ['Name', 'RGB']].tolist()
    return closestPalette


def newColorMP(pixelListLAB):
    with Pool(PROC) as pool:
        partial_func = partial(newColor, palette=palette_df)
        '''
        Trial with 9000 records. 
        Chunksize | Time Taken
                1 |      11.40
               10 |      12.37
               40 |      11.45
          ---> 50 |      11.20 <---
               60 |      11.60
              100 |      11.90
              500 |      13.47
             1000 |      13.31
        '''
        return list(tqdm(pool.imap(partial_func, pixelListLAB, chunksize=50),
                         desc='Finding Closest Color', total=len(unique_pixel_df),
                         mininterval=0.5))
#        return pool.map(partial_func, pixelListLAB, chunksize=50)


def create_canvas(b_w, b_h, c_r, c_c):
    canvas_width = b_w * c_c
    canvas_height = b_h * c_r
    return (canvas_width, canvas_height)


def create_image_grid(image):
    _pixel = image.load()
    for _i in range(0,image.size[0],PIXEL_SIZE):
      for _j in range(0,image.size[1],PIXEL_SIZE):
        for _r in range(PIXEL_SIZE):
          _pixel[_i+_r, _j] = BACKGROUND_COLOR
          _pixel[_i, _j+_r] = BACKGROUND_COLOR
    return image


def create_image(pixel_df, canvas_width, canvas_height):
    _pixel_list = pixel_df.values.tolist()
    _pixel_list = np.asarray(_pixel_list, dtype=np.uint8)
    
    _image = np.dstack((_pixel_list[:,0], _pixel_list[:,1], _pixel_list[:,2])).reshape(canvas_width, canvas_height, 3)
    _image = Image.fromarray(_image, 'RGB')
    _image = _image.resize((canvas_width * PIXEL_SIZE, canvas_height * PIXEL_SIZE), Image.NEAREST)
    return _image
    
    

BOARD_WIDTH = 50
BOARD_HEIGHT = 50

CANVAS_ROWS = 2
CANVAS_COLUMNS = 2

PIXEL_SIZE = 10
BACKGROUND_COLOR = (0,) * 3

PROC = os.cpu_count()

PIXEL_PER_BAG = 300

LOAD_DF = False

PALETTE = {"Black": (0,0,0),
           "Blue": (15,51,125),
           "Army Green": (15,29,15),
           "Green": (26,141,52),
           "Black Gray": (41,41,41),
           "Royal Blue": (21,33,55),
           "Jujube Red": (85,11,18),
           "Purple": (62,39,111),
           "Sky Blue": (14,95,161),
           "Light Green": (158,180,19),
           "Rose Red": (159,30,81),
           "Brown": (132,72,42),
           "Earthy Yellow": (166,141,95),
           "Dark Gray": (97,98,93),
           "Light Blue": (120,163,205),
           "Light Gray": (139,143,144),
           "Sand Yellow": (230,208,122),
           "Beige": (186,175,121),
           "Flesh": (247,214,193),
           "Red": (230,53,47),
           "Orange": (241,118,59),
           "Pink": (213,145,186),
           "Yellow": (255,229,30),
           "White": (255,255,255),
           "Khaki": (170, 162, 116)}

if __name__ == '__main__':
    
    canvas_width, canvas_height = create_canvas (BOARD_WIDTH, BOARD_HEIGHT, CANVAS_ROWS, CANVAS_COLUMNS)

    ###########################################################################
    ###########################################################################
    ###########################################################################    
    
    palette_df = pd.DataFrame()
    palette_df['Name'] = PALETTE.keys()
    palette_df['RGB']  = PALETTE.values()
    palette_df['sRGB'] = palette_df['RGB'].map(setRGB)
    palette_df['LAB']  = palette_df['sRGB'].map(convertLAB)
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    image = Image.open('lion.jpg')
    image = image.resize((canvas_width, canvas_height), Image.LANCZOS)
#    image = image.convert('P', palette=Image.ADAPTIVE, colors=256).convert('RGB')
    
    ###########################################################################
    ###########################################################################
    #Filter     Downscaling quality     Upscaling quality     Performance
    #NEAREST	 	 	                                      ⭐⭐⭐⭐⭐
    #BOX	    ⭐	 	                                      ⭐⭐⭐⭐
    #BILINEAR	⭐	                    ⭐	                  ⭐⭐⭐
    #HAMMING	⭐⭐	 	                                  ⭐⭐⭐
    #BICUBIC	⭐⭐⭐	                ⭐⭐⭐	              ⭐⭐
    #LANCZOS	⭐⭐⭐⭐	            ⭐⭐⭐⭐	          ⭐
    ###########################################################################
    ########################################################################### 
    
    pixel_df = pd.DataFrame()
    pixel_array = np.array(image.getdata())
    pixel_df['RGB'] = list(zip(pixel_array[:,0], pixel_array[:,1], pixel_array[:,2]))
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    if LOAD_DF:
        unique_pixel_df = pd.read_hdf('unique_pixel.h5', 'unique_pixel_df')
    else:
        unique_pixel_df = pd.DataFrame()
        unique_pixel_df['RGB'] = pixel_df['RGB'].unique().tolist()    
        unique_pixel_df['sRGB'] = unique_pixel_df['RGB'].map(setRGB)  
        unique_pixel_df['LAB'] = unique_pixel_df['sRGB'].map(convertLAB)
        
    ###########################################################################
    ###########################################################################
    ###########################################################################

        if len(unique_pixel_df) > 1000:
            print(f'Starting multiprocessing with {PROC} processes.')
            _t_mp = time()
            unique_pixel_df['newName'], unique_pixel_df['newRGB'] = zip(*newColorMP(unique_pixel_df['LAB']))   
            print(f'MP took {time()-_t_mp:.3f} seconds.')
        else:
            print(f'Starting sequential processing.')
            tqdm.pandas(desc='Finding Closest Color')    
            _t_seq = time()
            func = partial(newColor, palette=palette_df)
            unique_pixel_df['newName'], unique_pixel_df['newRGB'] = zip(*unique_pixel_df['LAB'].progress_map(func))
            print(f'Seq took {time()-_t_seq:.3f} seconds.')
        
        unique_pixel_df.to_hdf('unique_pixel.h5', 'unique_pixel_df')

    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    _t_re = time()
    pixel_df['newRGB'] = pixel_df['RGB'].map(unique_pixel_df.set_index('RGB')['newRGB'])
    pixel_df['newName'] = pixel_df['RGB'].map(unique_pixel_df.set_index('RGB')['newName'])
    print(f'Replace took {time()-_t_re:.3f} seconds.')
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    unique_pixel_df.to_hdf('unique_pixel.h5', 'unique_pixel_df')
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    bags_of_color = pd.DataFrame()
    bags_of_color['Occurences'] = pixel_df['newName'].value_counts()
    assert len(pixel_df) == bags_of_color['Occurences'].sum(), 'Occurence not equal to pixel count.'
    bags_of_color['Bags'] = bags_of_color['Occurences'].map(lambda Occ: ceil(Occ/PIXEL_PER_BAG))
    pprint(bags_of_color)

    ###########################################################################
    ###########################################################################
    ###########################################################################
    
    new_image = create_image(pixel_df['newRGB'], canvas_width, canvas_height)
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
         
    new_image = create_image_grid(new_image)
    
    #newImage.save(imageName + ' - pixelated.png', format = 'PNG')
    new_image.save(f'pixalated - {time()}.png', format = 'PNG')
    image.show()
    new_image.show()
    