import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def equalize_int16(img16s: np.ndarray) -> np.ndarray:
    """
    Perform histogram equalization directly in the signed int16 domain.
    Input:
      img16s:   H×W array of dtype=np.int16/uint16
    Output:
      eq16:     H×W array of dtype=np.uint16 where the histogram has been
                spread across the full uint16 range.
    """
    # 1) Flatten and offset so that pixel values sit in [0…65535]
    flat = img16s.ravel().astype(np.int32)  # use int32 to avoid overflow
    offset = flat + 32768  # now in [0…65535]

    # 2) Compute a 65536-bin histogram over [0…65535]
    #    (We want one bin per possible int16 value.)
    hist = np.bincount(offset, minlength=65536)
    #    total number of pixels:
    total_pixels = offset.size

    # 3) Compute the cumulative distribution (CDF)
    cdf = hist.cumsum().astype(np.float64)

    #    Find CDF_min = first nonzero CDF entry (to avoid mapping noise to zero)
    #    i.e. we ignore empty bins at the very low end of the offset range.
    nonzero = np.nonzero(cdf)[0]
    if nonzero.size == 0:
        # image is constant zero → nothing to do
        return img16s.copy()
    cdf_min = cdf[nonzero[0]]

    # 4) Build a LUT so that each offset‐value v maps to:
    #      round( (cdf[v] - cdf_min) / (total_pixels - cdf_min) * 65535 )
    denom = total_pixels - cdf_min
    # Avoid division by zero if image is flat:
    if denom <= 0:
        # All pixels have exactly the same value → return a flat image
        return img16s.copy()

    # Normalize CDF into [0…1]
    cdf_normalized = (cdf - cdf_min) / denom
    # Clip to [0,1] in case of tiny negative floats:
    cdf_normalized = np.clip(cdf_normalized, 0.0, 1.0)

    # Build LUT: new_offset = floor( cdf_normalized * 65535 )
    lut = np.floor(cdf_normalized * 65535.0).astype(np.uint16)

    # 5) Apply LUT back to every pixel, then subtract 32768 to get signed int16
    equalized_offset = lut[offset]  # shape=(H*W,), dtype=uint16
    equalized16 = equalized_offset.reshape(img16s.shape).astype(np.uint16)

    return equalized16


def remove_vignette(image, blur_str: int, sigma_str: int = None):
    if sigma_str is None:
        sigma_str = blur_str // 2

    lf_image = cv2.GaussianBlur(image, (blur_str, blur_str), sigma_str)
    lf_image = image - lf_image

    return lf_image


def blur_img(image, blur_shape=(1, 31)):
    blurred = cv2.blur(image, blur_shape)
    return blurred


def read_h5_images_to_df(h5_path):
    """
    Read all uint16 image datasets from an HDF5 file into a DataFrame.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'name': str, dataset path with '/' → '_'
        - 'matrix': np.ndarray, the uint16 image data
    """
    names = []
    matrices = []

    with h5py.File(h5_path, 'r') as h5f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.dtype == np.uint16:
                # record a sanitized name and the array
                names.append(name.replace('/', '_'))
                matrices.append(obj[()])

        h5f.visititems(visitor)

    df = pd.DataFrame({
        'name': names,
        'matrix': matrices
    })
    return df


class Reader:
    def __init__(self, path_to_file: str):
        self.dataframe = read_h5_images_to_df(path_to_file)

    def preprocess_width(self, debug=False):

        def squeeze_image(img):
            H, W = img.shape
            img_squeezed = img.reshape(H, 4, W // 4).sum(axis=1)
            img_squeezed = cv2.normalize(img_squeezed, None, 0, 65536, cv2.NORM_MINMAX, cv2.CV_16U)
            return img_squeezed

        self.dataframe['matrix'] = self.dataframe['matrix'].apply(squeeze_image)

        if debug:
            _, image = self[1]
            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.show()

        pass

    def convert_to_uint8(self):
        pass

    def equalize_images(self, debug=False):
        self.dataframe['matrix'] = self.dataframe['matrix'].apply(equalize_int16)

        if debug:
            _, image = self[1]
            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.show()
            plt.figure()
            plt.hist(image.flatten(), bins=256)
            plt.show()

    def remove_vignette(self, blur, sigma=None, debug=False):

        # structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (blur, blur))
        _, img = self[1]

        lf_img = remove_vignette(img, blur, sigma)

        if debug:
            plt.figure()
            plt.imshow(lf_img, cmap='gray')
            plt.show()

    def __getitem__(self, item):
        return self.dataframe.iloc[item]['name'], self.dataframe.iloc[item]['matrix']

    def __len__(self):
        return len(self.dataframe)

    def find_index_by_name(self, namesko):
        return self.dataframe.index[self.dataframe['name'] == namesko][0]

    def show_image(self, item):
        plt.imshow(self.dataframe.iloc[item]['matrix'], cmap='gray')
        plt.title(self.dataframe.iloc[item]['name'])
        plt.tight_layout()
        plt.show()

    def return_data(self):
        return self.dataframe
