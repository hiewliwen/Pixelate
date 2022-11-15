import numpy as np
from utils import Pixelate_Image


# Temporary Workaround
# https://github.com/gtaylor/python-colormath/issues/104#issuecomment-1247481638
if np.__version__ > "1.16":
    def patch_asscalar(a):
        return a.item()
setattr(np, "asscalar", patch_asscalar)


PALETTE = {"Black": (0, 0, 0),
           "Blue": (15, 51, 125),
           "Army Green": (15, 29, 15),
           "Green": (26, 141, 52),
           "Black Gray": (41, 41, 41),
           "Royal Blue": (21, 33, 55),
           "Jujube Red": (85, 11, 18),
           "Purple": (62, 39, 111),
           "Sky Blue": (14, 95, 161),
           "Light Green": (158, 180, 19),
           "Rose Red": (159, 30, 81),
           "Brown": (132, 72, 42),
           "Earthy Yellow": (166, 141, 95),
           "Dark Gray": (97, 98, 93),
           "Light Blue": (120, 163, 205),
           "Light Gray": (139, 143, 144),
           "Sand Yellow": (230, 208, 122),
           "Beige": (186, 175, 121),
           "Flesh": (247, 214, 193),
           "Red": (230, 53, 47),
           "Orange": (241, 118, 59),
           "Pink": (213, 145, 186),
           "Yellow": (255, 229, 30),
           "White": (255, 255, 255),
           "Khaki": (170, 162, 116)}

BOARD_WIDTH = 50
BOARD_HEIGHT = 50

CANVAS_ROWS = 2
CANVAS_COLUMNS = 2

PIXEL_SIZE = 10
BACKGROUND_COLOR = (0,) * 3

PIXELS_PER_BAG = 300

IMG_PATH = 'lion.jpg'
OUT_PATH = 'pixelated lion.jpg'

if __name__ == '__main__':

    pixel_art = Pixelate_Image(img_path=IMG_PATH, save_path = OUT_PATH, palette=PALETTE, 
                               board_width=BOARD_WIDTH, board_height=BOARD_HEIGHT, 
                               canvas_rows= CANVAS_ROWS, canvas_columns=CANVAS_COLUMNS, 
                               pixels_per_bag=PIXELS_PER_BAG, bg_color=BACKGROUND_COLOR, 
                               save_df=False)

    
    new_image = pixel_art.pixelize()
    pixel_art.bags_of_colors()