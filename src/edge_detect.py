import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from skimage.transform import radon


def calc_hist(matrix):
    """Sum each column to get bin values.
    Parameters
    ----------
    matrix : np.ndarray
        2D grayscale image.

    """
    return np.sum(matrix, axis=0)


def canny_edges(img_uint8, blur_kernel=(5, 5), sigma=1.5, edge_width=3, sigma_median=0.33, min_low=10, min_high=25):
    """
    Apply Gaussian blur and Canny edge detection with adaptive thresholds.

    Parameters
    ----------
    img_uint8 : np.ndarray
        2D grayscale image: uint8
    blur_kernel : tuple of int, optional
        Kernel size for Gaussian blur. Default is (5, 5).
    sigma : float, optional
        Standard deviation for Gaussian kernel. Default is 1.5.
    edge_width : int, optional
        Desired width of the thinned edges. Default is 2.
    sigma_median : float, optional
        Multiplier for the median value. Default is 0.33.
    min_low : int, optional
        Lower threshold for Canny edge detection. Default is 10.
    min_high : int, optional
        Upper threshold for Canny edge detection. Default is 30.

    Returns
    -------
    edges : np.ndarray
        Binary image (uint8) with detected edges (0 or 255).

    """
    blurred = cv.GaussianBlur(img_uint8, blur_kernel, sigmaX=sigma)
    v = np.median(blurred)

    auto_low = int(max(0, (1.0 - sigma_median) * v))
    auto_high = int(min(255, (1.0 + sigma_median) * v))
    th_low = max(auto_low, min_low)
    th_high = max(auto_high, min_high)

    edges = cv.Canny(blurred, threshold1=th_low, threshold2=th_high)
    if edge_width < 3:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv.erode(edges, kernel, iterations=3 - edge_width)
    return edges


def compute_vertical_profile(edge_image):
    """Compute vertical edge profile by summing edge pixels in each column.

    Parameters
    ----------
    edge_image : np.ndarray
        Binary edge image (output from Canny).
    Returns
    -------
    profile : np.ndarray
        1D array containing the sum of edge pixels per column.

    """
    return edge_image.sum(axis=0)


def detect_vertical_defects_local(profile, min_distance=20, prominence_factor=1000.0):
    """Detect vertical defect positions based on local prominence.

    Parameters
    ----------
    profile : np.ndarray
        1D array with vertical edge sums per column.
    min_distance : int, optional
        Minimum distance (in pixels) between detected peaks. Default is 20.

    Returns
    -------
    peaks : np.ndarray
        Array of column indices corresponding to vertical defect candidates.

    """
    peaks, properties = find_peaks(profile, distance=min_distance)
    if len(peaks) == 0:
        return np.array([])

    validated_peaks = []
    for peak in peaks:
        window_start = max(0, peak - 50)
        window_end = min(len(profile), peak + 51)
        local_profile = profile[window_start:window_end]
        local_median = np.median(local_profile)
        if profile[peak] > local_median * prominence_factor:
            validated_peaks.append(peak)

    return np.array(validated_peaks)


def filter_edges_by_length_and_direction(edges, min_length=3, max_direction_change=90):
    """Filter edge contours based on minimal length and maximum angle.

    Parameters
    ----------
    edges : np.ndarray
        Binary edge image (output from Canny).
    min_length : int, optional
        Minimum length of detected edges. Default is 3.

    max_direction_change : int, optional
        Maximum angle difference between edges. Default is 90 degrees.

    Returns
    -------
    filtered_edges : np.ndarray
      Binary image with filtered edges.
    """
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    filtered_edges = np.zeros_like(edges)
    for cnt in contours:
        if len(cnt) >= min_length:
            pts = cnt.squeeze()

            directions = np.diff(pts, axis=0)
            angles = np.arctan2(directions[:, 1], directions[:, 0]) * 180 / np.pi

            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)

            max_diff = np.max(angle_diffs)

            if max_diff <= max_direction_change:
                cv.drawContours(filtered_edges, [cnt], -1, 255, 1)

    return filtered_edges


class EdgeDetector:
    def __init__(self, image):
        self.defect_canny = None
        self.defect_histogram = None
        self.defect_both = None
        self.histogram_threshold = None
        self.histogram_histogram = None

        norm_img = (image - image.min()) / (image.max() - image.min())
        self.img_uint8 = (norm_img * 255).astype(np.uint8)

    def detect_edges_canny(self, blur_kernel=(5, 5), sigma=1.5, percentile=98.0, min_distance=4,
                           min_length=3, max_angle_change=90, roc=False):
        edges = canny_edges(self.img_uint8, blur_kernel, sigma)
        edges = filter_edges_by_length_and_direction(edges, min_length, max_angle_change)
        vertical_profile = compute_vertical_profile(edges)

        filtered_profile = vertical_profile.copy()
        if np.any(filtered_profile > 0):
            if not roc:
                min_edge_count = np.percentile(filtered_profile[filtered_profile > 0], 80)
                filtered_profile[filtered_profile < min_edge_count] = 0

            if np.sum(filtered_profile) == 0:
                return np.array([])

            threshold = np.percentile(filtered_profile[filtered_profile > 0], percentile)
            defect_columns, _ = find_peaks(filtered_profile, height=threshold, distance=min_distance)
        else:
            defect_columns = np.array([])

        self.defect_canny = defect_columns + 1
        return defect_columns

    def detect_edges_histogram(self, threshold_value=1.4, min_distance=4):
        column_sums = calc_hist(self.img_uint8)
        threshold = threshold_value * np.mean(column_sums)
        peaks, properties = find_peaks(column_sums, height=threshold, distance=min_distance)

        self.histogram_threshold = threshold
        self.histogram_histogram = column_sums
        self.defect_histogram = peaks
        return peaks

    def detect_edges_radon(self, threshold=1.4, debug=False):
        # tophat
        img = self.img_uint8.astype(np.float32)
        sobelx = cv.Sobel(img, ddepth=cv.CV_32F, dx=1, dy=0, ksize=3)
        to_analyze = np.abs(sobelx)

        sinogram = radon(to_analyze, theta=np.array([0.]), circle=False)

        col_profile = sinogram[:, 0]

        raw = img
        H_img, W_img = raw.shape
        offset = int((col_profile.shape[0] - W_img) // 2)
        filtered_cols = col_profile[offset:offset + W_img]

        peaks, _ = find_peaks(filtered_cols, height=(filtered_cols.mean() * threshold), distance=4)
        # `peaks` now holds the x‐positions of candidate vertical stripes.

        # 4) Convert “peak index” to actual image‐column index
        #    Depending on how radon zero‐pads, your “column index” might be shifted. Typically:
        #    column_index = peaks[i] - (H - W) // 2

        # 5) Visualize:
        if debug:

            plt.figure(figsize=(6, 6))
            plt.plot(filtered_cols)
            plt.show()

            overlay = cv.cvtColor(np.clip(raw, 0, 255).astype(np.uint8), cv.COLOR_GRAY2BGR)
            for x in peaks:
                cv.line(overlay, (x, 0), (x, H_img - 1), (0, 255, 0), 1)

            plt.figure(figsize=(6, 6))
            plt.imshow(overlay[:, :, ::-1])
            plt.title("Radon‐based Detected Vertical Columns")
            plt.axis('off')
            plt.show()

            plt.figure(figsize=(6, 6))
            plt.plot(col_profile)
            plt.show()

        return peaks

    def detect_defects(self, method, **kwargs):

        if kwargs.get('method_threshold', None) is not None:
            normal_threshold = kwargs.get('method_threshold')
        else:
            normal_threshold = 1.4

        if method == "canny":
            if kwargs.get("canny_percentile", None) is not None:
                canny_percentile = kwargs.get("canny_percentile")
            else:
                canny_percentile = 95.
            return self.detect_edges_canny(percentile=canny_percentile)

        elif method == "histogram":
            return self.detect_edges_histogram(normal_threshold)
        elif method == "radon":
            return self.detect_edges_radon(normal_threshold)
        else:
            raise ValueError("Wrong detection_method selected. It must be 'canny', 'histogram' or 'radon'")

    def detect_columns(self):
        if self.defect_canny is None:
            self.detect_edges_canny()
        if self.defect_histogram is None:
            self.detect_edges_histogram()
        self.defect_both = np.intersect1d(self.defect_canny, self.defect_histogram)
        return self.defect_both

    def plot_histogram(self):
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(self.img_uint8)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.bar(np.arange(len(self.histogram_histogram)), self.histogram_histogram)
        plt.hlines(y=self.histogram_threshold, xmin=0, xmax=len(self.histogram_histogram), color='tab:red')
        plt.show()

    def plot_detected_defects(self):
        plt.figure(figsize=(6, 8))

        plt.subplot(2, 1, 1)
        plt.imshow(self.img_uint8, cmap='gray')

        if self.defect_canny is not None and len(self.defect_canny) > 0:
            for x in self.defect_canny:
                plt.axvline(x=x, color='green', linestyle='--', linewidth=1, alpha=0.9)

        if self.defect_histogram is not None and len(self.defect_histogram) > 0:
            for x in self.defect_histogram:
                plt.axvline(x=x, color='red', linestyle='-', linewidth=1, alpha=0.7)

        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.imshow(self.img_uint8, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
