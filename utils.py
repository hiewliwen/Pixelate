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
from typing import Optional, Union

# Temporary Workaround
# https://github.com/gtaylor/python-colormath/issues/104#issuecomment-1247481638
if np.__version__ > "1.16":
    def patch_asscalar(a):
        """
        Numpy.asscalar is deprecated since 1.16. Colormath had not updated their code. This is a workaround.

        Parameters
        ----------
        a : numpy.asscalar
            numpy.asscalar object.

        Returns
        -------
        scalar
            scalar object.

        """
        return a.item()
setattr(np, "asscalar", patch_asscalar)


class Pixelate_Image():
    def __init__(self,
                 img_path: str, save_path: str, palette: dict, board_width: int, board_height: int,
                 canvas_rows: Optional[int] = 1, canvas_columns: Optional[int] = 1,
                 load_h5: Optional[bool] = False, save_df: Optional[bool] = True,
                 h5_fp: Optional[str] = 'unique_pixel.h5', table: Optional[str] = 'unique_pixel_df',
                 pixel_size: Optional[int] = 10, pixels_per_bag: Optional[int] = None,
                 bg_color: Optional[list] = (0, 0, 0), grid: Optional[bool] = True,
                 save_img: Optional[bool] = True):
        """
        Initialize Pixelate_Image Class.

        Parameters
        ----------
        img_path : str
            DESCRIPTION.
        save_path : str
            DESCRIPTION.
        palette : dict
            DESCRIPTION.
        board_width : int
            DESCRIPTION.
        board_height : int
            DESCRIPTION.
        canvas_rows : Optional[int], optional
            DESCRIPTION. The default is 1.
        canvas_columns : Optional[int], optional
            DESCRIPTION. The default is 1.
        load_h5 : Optional[bool], optional
            DESCRIPTION. The default is False.
        save_df : Optional[bool], optional
            DESCRIPTION. The default is True.
        h5_fp : Optional[str], optional
            DESCRIPTION. The default is 'unique_pixel.h5'.
        table : Optional[str], optional
            DESCRIPTION. The default is 'unique_pixel_df'.
        pixel_size : Optional[int], optional
            DESCRIPTION. The default is 10.
        pixels_per_bag : Optional[int], optional
            DESCRIPTION. The default is None.
        bg_color : Optional[list], optional
            DESCRIPTION. The default is (0, 0, 0).
        grid : Optional[bool], optional
            DESCRIPTION. The default is True.
        save_img : Optional[bool], optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        self.img_path = img_path
        self.save_path = save_path
        self.palette = palette
        self.board_width = board_width
        self.board_height = board_height
        self.canvas_rows = canvas_rows
        self.canvas_columns = canvas_columns
        self.load_h5 = load_h5
        self.save_df = save_df
        self.h5_fp = h5_fp
        self.table = table
        self.pixel_size = pixel_size
        self.pixels_per_bag = pixels_per_bag
        self.bg_color = bg_color
        self.grid = grid
        self.save_img = save_img
        self.proc = os.cpu_count()

    def setRGB(self, RGB: Union[list, tuple]) -> sRGBColor:
        """
        Create sRGB object form RGB.

        Parameters
        ----------
        RGB : tuple or list
            Pixel values of R, G, B [0-255].

        Returns
        -------
        sRGBColor
            sRGB representation of RGB.

        """
        R, G, B = RGB
        self.sRGB = sRGBColor(R, G, B, is_upscaled=True)
        return self.sRGB

    def convertLAB(self, sRGB: sRGBColor) -> LabColor:
        """
        Convert sRGB object into L*a*b object.

        Parameters
        ----------
        sRGB : sRGBColor
            sRGB object.

        Returns
        -------
        LabColor
            L*a*b representation of sRGB object.

        """
        self.LAB = convert_color(sRGB, LabColor)
        return self.LAB

    def create_canvas(self) -> tuple:
        """
        Create the canvas.

        Returns
        -------
        tuple
            Canvas width, Canvas height.

        """
        self.canvas_width = self.board_width * self.canvas_columns
        self.canvas_height = self.board_height * self.canvas_rows
        return (self.canvas_width, self.canvas_height)

    def create_palette_df(self) -> pd.DataFrame:
        """
        Create a DataFrame for the palette.

        Returns
        -------
        DataFrame
            DataFrame of palette in various color spaces.

        """
        self.palette_df = pd.DataFrame()
        self.palette_df['Name'] = self.palette.keys()
        self.palette_df['RGB'] = self.palette.values()
        self.palette_df['sRGB'] = self.palette_df['RGB'].apply(self.setRGB)
        self.palette_df['LAB'] = self.palette_df['sRGB'].apply(self.convertLAB)
        return self.palette_df

    def create_pixel_df(self) -> pd.DataFrame:
        """
        Create a DataFrame with input image.

        Returns
        -------
        DataFrame
            DataFrame of input image.

        #######################################################################
        #Filter     Downscaling quality     Upscaling quality     Performance
        #NEAREST                                                    ⭐⭐⭐⭐⭐
        #BOX        ⭐                                               ⭐⭐⭐⭐
        #BILINEAR    ⭐                        ⭐                      ⭐⭐⭐
        #HAMMING    ⭐⭐                                           ⭐⭐⭐
        #BICUBIC    ⭐⭐⭐                    ⭐⭐⭐                  ⭐⭐
        #LANCZOS    ⭐⭐⭐⭐                ⭐⭐⭐⭐              ⭐
        #######################################################################

        """
        self.image = Image.open(self.img_path)
        self.image = self.image.resize(
            (self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)

        self.pixel_df = pd.DataFrame()
        self.pixel_array = np.array(self.image.getdata())
        self.pixel_df['RGB'] = list(
            zip(self.pixel_array[:, 0], self.pixel_array[:, 1], self.pixel_array[:, 2]))

        return self.pixel_df

    def create_unique_pixel_df(self) -> pd.DataFrame:
        """
        Create a DataFrame of unique pixels of the image.

        Parameters
        ----------
        load_h5 : bool, optional
            If True, load existing dataframe. The default is False.
        h5_fp : str, optional
            File path of h5 file to be loaded. The default is 'unique_pixel.h5'.
        table : str, optional
            Table name in the h5. The default is 'unique_pixel_df'.

        Returns
        -------
        DataFrame
            DataFrame of unique pixels of the image.

        """
        if self.load_h5:
            self.unique_pixel_df = pd.read_hdf(self.h5_fp, self.table)
        else:
            self.unique_pixel_df = pd.DataFrame()
            self.unique_pixel_df['RGB'] = self.pixel_df['RGB'].unique(
            ).tolist()
            self.unique_pixel_df['sRGB'] = self.unique_pixel_df['RGB'].apply(
                self.setRGB)
            self.unique_pixel_df['LAB'] = self.unique_pixel_df['sRGB'].apply(
                self.convertLAB)

        return self.unique_pixel_df

    def get_closest_colors(self) -> pd.DataFrame:
        """
        Get the closest colour from the list of palette.

        Parameters
        ----------
        save_df : bool, optional
            If True, save dataframe into h5. The default is True.
        h5_fp : str, optional
            File path of h5 file to be saved to. The default is 'unique_pixel.h5'.
        table : str, optional
            Table name in the h5. The default is 'unique_pixel_df'.

        Returns
        -------
        DataFrame
            DataFrame of unique pixels of the image..

        """
        if len(self.unique_pixel_df) > 1_000:
            print(f'Starting multiprocessing with {self.proc} processes.')
            _t_mp = time()
            self.unique_pixel_df['newName'], self.unique_pixel_df['newRGB'] = zip(
                *self.newColorMP(self.unique_pixel_df['LAB']))
            print(f'MP took {time()-_t_mp:.3f} seconds.')
        else:
            print('Starting sequential processing.')
            tqdm.pandas(desc='Finding Closest Color')
            _t_seq = time()
            func = partial(self.newColor, palette=self.palette_df)
            self.unique_pixel_df['newName'], self.unique_pixel_df['newRGB'] = zip(
                *self.unique_pixel_df['LAB'].progress_map(func))
            print(f'Seq took {time()-_t_seq:.3f} seconds.')

        if self.save_df:
            self.unique_pixel_df.to_hdf(self.h5_fp, self.table)

        return self.unique_pixel_df

    def deltaE(self, testColorLAB: LabColor, paletteLAB: pd.DataFrame) -> np.asscalar:
        """
        Calculate deltaE between 2 L*a*b.

        Parameters
        ----------
        testColorLAB : LabColor
            Color to test.
        paletteLAB : pd.DataFrame
            Copy of palette DataFrame.

        Returns
        -------
        deltaE : Numpy AsScalar
            deltaE result.

        """
        deltaE = abs(delta_e_cie2000(testColorLAB, paletteLAB))
        return deltaE

    def newColor(self, pixelLAB: LabColor) -> LabColor:
        """
        Calculate the new color.

        Parameters
        ----------
        pixelLAB : LabColor
            Original color in LabColor.

        Returns
        -------
        LabColor
            New color in LabColor.

        """
        paletteCopy = self.palette_df[['Name', 'RGB', 'LAB']].copy()
        paletteCopy['deltaE'] = paletteCopy['LAB'].map(
            lambda pLAB: self.deltaE(pixelLAB, pLAB))
        minDeltaEIDX = paletteCopy['deltaE'].idxmin(axis=0)
        closestPalette = paletteCopy.loc[paletteCopy.index[minDeltaEIDX], [
            'Name', 'RGB']].tolist()
        return closestPalette

    def newColorMP(self, pixelListLAB: pd.Series) -> pd.Series:
        """
        Calculate the new color with multiprocessing.

        Parameters
        ----------
        pixelListLAB : pd.Series
            Series of pixels in L*a*b.

        Returns
        -------
        pd.Series
            New color.

        """
        with Pool(self.proc) as pool:

            return list(tqdm(pool.imap(self.newColor, pixelListLAB, chunksize=200),
                             desc='Finding Closest Color', total=len(self.unique_pixel_df),
                             mininterval=0.5))

    def replace_color(self) -> pd.DataFrame:
        """
        Replace pixels in image with new color.

        Returns
        -------
        pixel_df : DataFrame
            DataFrame of image.

        """
        t_re = time()
        self.pixel_df['newRGB'] = self.pixel_df['RGB'].map(
            self.unique_pixel_df.set_index('RGB')['newRGB'])
        self.pixel_df['newName'] = self.pixel_df['RGB'].map(
            self.unique_pixel_df.set_index('RGB')['newName'])
        print(f'Replace took {time()-t_re:.3f} seconds.')

        return self.pixel_df

    def create_image(self) -> Image:
        """
        Create a new image from pixel_df.

        Parameters
        ----------
        grid : bool, optional
            If True, add grid lines to the image. The default is True.
        save_img : bool, optional
            If True, save the final image. The default is True.

        Returns
        -------
        Image
            Pixelated image.

        """
        pixel_list = self.pixel_df['newRGB'].values.tolist()
        pixel_list = np.asarray(pixel_list, dtype=np.uint8)

        new_image = np.dstack((pixel_list[:, 0], pixel_list[:, 1], pixel_list[:, 2])).reshape(
            self.canvas_width, self.canvas_height, 3)
        new_image = Image.fromarray(new_image, 'RGB')
        new_image = new_image.resize(
            (self.canvas_width * self.pixel_size, self.canvas_height * self.pixel_size), Image.NEAREST)

        if self.grid:
            new_image = self.create_image_grid(new_image)

        if self.save_img:
            # new_image.save(f'pixalated - {time()}.png', format='PNG')
            new_image.save(self.save_path)

        return new_image

    def create_image_grid(self, image: Image) -> Image:
        """
        Create grid around each pixel.

        Parameters
        ----------
        image : Image
            Image object create in "create_image".

        Returns
        -------
        Image
            Image object with grid superimposed.

        """
        _pixel = image.load()
        for _i in range(0, image.size[0], self.pixel_size):
            for _j in range(0, image.size[1], self.pixel_size):
                for _r in range(self.pixel_size):
                    _pixel[_i+_r, _j] = self.bg_color
                    _pixel[_i, _j+_r] = self.bg_color
        return image

    def bags_of_colors(self) -> pd.DataFrame:
        """
        Calculate the number of bags for each pixels.

        Returns
        -------
        bags : DataFrame
            Colour and number of bags.

        """
        bags = pd.DataFrame()
        bags['Occurences'] = self.pixel_df['newName'].value_counts()
        assert len(self.pixel_df) == bags['Occurences'].sum(
        ), 'Occurence not equal to pixel count.'
        bags['Bags'] = bags['Occurences'].map(
            lambda Occ: ceil(Occ/self.pixels_per_bag))
        pprint(bags)

        return bags

    def pixelize(self, show_image: Optional[bool] = True) -> Image:
        """
        Pixelate the input image.

        Parameters
        ----------
        show_image : bool, optional
            If True, show the pixelated image. The default is True.

        Returns
        -------
        Image
            Pixelated image.

        """
        # Create the canvas
        self.create_canvas()

        # Create a DataFrame for the palette
        self.create_palette_df()

        # Create a DataFrame with input image
        self.create_pixel_df()

        # Create a DataFrame of unique pixels of the image.
        self.create_unique_pixel_df()

        # Get the closest colors for each unique pixels
        self.get_closest_colors()

        # Replace pixel_df with new colors
        self.replace_color()

        # Generate new image
        image = self.create_image()

        if show_image:
            image.show()
            
        return image