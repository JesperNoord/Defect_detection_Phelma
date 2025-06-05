import copy
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import cm
from matplotlib.patches import Rectangle
from moviepy import ImageSequenceClip

from src.reading import Reader
from src.type_detect import EdgeProcessor


class Coldect:

    def __init__(self, settings_path):
        self.original_data = None
        self.preprocessed_data = None
        self.columns = None
        self.defect_data = None
        self.processor = None
        self.detection_data = None
        self.annotated_data = None

        # load settings
        with open(settings_path) as settings_file:
            self.settings = yaml.load(settings_file, Loader=yaml.SafeLoader)

    def load_file(self, path_to_file):
        self.original_data = Reader(path_to_file)
        self.preprocessed_data = copy.deepcopy(self.original_data)

    def preprocess(self, debug=False):
        out_data = copy.deepcopy(self.original_data)

        if self.settings['columns_per_amplifier'] == 4:
            out_data.preprocess_width()

        vignette_settings = self.settings['remove_vignette']
        if vignette_settings is []:
            vignette_settings = [51]

        preprocessing_parameters = {
            'equalize_images': [],
            'remove_vignette': vignette_settings
        }

        preprocessing_settings = {
            'equalize_images': out_data.equalize_images,
            'remove_vignette': out_data.remove_vignette,
        }

        for setting in self.settings['preprocessing_methods']:
            fcn = preprocessing_settings[setting]
            parameter = preprocessing_parameters[setting]
            fcn(*parameter)

        self.preprocessed_data = out_data

        if debug:
            _, img_in = self.original_data[0]
            _, img_out = out_data[0]

            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.imshow(img_in, cmap='gray')
            plt.title('Input Image')
            plt.subplot(122)
            plt.imshow(img_out, cmap='gray')
            plt.title('Preprocessed Image')
            plt.tight_layout()
            plt.show()

    def detect_defects(self):
        in_data = self.preprocessed_data
        self.processor = EdgeProcessor(in_data, self.settings)
        print(self.processor.df)
        fragmented_data = self.processor.detect_fragmented()
        _, detection_data = self.processor.detect_blinking_or_noisy(fragmented_data)
        self.detection_data = detection_data
        return detection_data

    def detect_intensity(self):
        if self.detection_data is None:
            self.detect_defects()

        annotated = self.processor.append_intensity_metrics(self.detection_data)
        self.annotated_data = annotated
        return annotated

    def draw_detections_on_frame(
            self,
            frame_name: str,
            detections_df: pd.DataFrame = None,
            metric_col: str = "impact_score",
            box_width: int = 4,
            cmap="RdYlGn_r"
    ):
        """
        Visualise all detections for *frame_name*.

        Parameters
        ----------
        frame_name : str
            e.g. "Image 0042".
        detections_df : pd.DataFrame, default None
            If none, the detections from self.annotated_data are used.
            Table with one row per detection (all frames).  Must contain:
                • frame
                • position
                • status              ("BLINKING" / "NOISY")
                • col_has_fragmented  (bool)
                • *metric_col*        (float) – e.g. impact_score
        metric_col : str, default "impact_score"
            Column whose value drives the green-to-red colour scale.
        box_width : int, default 4
            Half-width in pixels of the coloured rectangle
            (actual width = 2·box_width + 1).
        cmap : str or Colormap, default "RdYlGn_r"
            Any matplotlib colormap.  Default is green→yellow→red.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Ready to be saved or added to an animation.
        """

        # ─── pick out this frame’s rows ────────────────────────────────
        if detections_df is None:
            if hasattr(self, "annotated_data") and self.annotated_data is not None:
                detections_df = self.annotated_data
            else:
                detections_df = self.detect_intensity()

        frame_mask = detections_df["frame"] == frame_name
        if not frame_mask.any():
            raise ValueError(f"No detections for frame '{frame_name}'")

        df_frame = detections_df.loc[frame_mask].copy()

        # ─── fetch the image -------------------------------------------------
        idx = self.preprocessed_data.find_index_by_name(frame_name)
        _, img = self.preprocessed_data[idx]  # img shape = H×W
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # ─── normalise the metric to [0, 1] for colour mapping -------------
        metric_vals = df_frame[metric_col].values.astype(float)
        vmin, vmax = metric_vals.min(), metric_vals.max()
        if np.isclose(vmin, vmax):
            norm_metric = np.zeros_like(metric_vals)  # all same colour
        else:
            norm_metric = (metric_vals - vmin) / (vmax - vmin)

        colormap = cm.get_cmap(cmap)

        # ─── prepare the canvas --------------------------------------------
        dpi = 100
        fig_w = img.shape[1] / dpi
        fig_h = img.shape[0] / dpi

        fig, ax = plt.subplots(
            figsize=(fig_w, fig_h),
            dpi=dpi,
            frameon=False  # no figure border
        )

        # Make the whole background black, not white
        fig.patch.set_facecolor("black")

        # Fill the *entire* figure with one axes
        ax.set_position([0, 0, 1, 1])  # left, bottom, width, height (in figure frac)

        ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_axis_off()

        # ─── draw bounding boxes & labels (unchanged) -----------------------
        H, W = img.shape
        half_w = box_width

        # ─── draw bounding boxes & labels -----------------------------------
        H, W = img.shape
        half_w = box_width

        for (_, row), norm in zip(df_frame.iterrows(), norm_metric):
            x = int(row["position"])
            colour = colormap(norm)

            # 1) coloured rectangle spanning full height
            rect = Rectangle(
                (x - half_w, 0),  # (x0, y0)
                width=half_w * 2 + 1,
                height=H,
                linewidth=1.2,
                edgecolor=colour,
                facecolor="none"
            )
            ax.add_patch(rect)

            # 2) label
            if row["status"] == "NOISY":
                label = "noisy"
            else:
                label = "fragmented blinking" if row["col_has_fragmented"] else "blinking"
            label += ', intensity={:.1f}'.format(row[metric_col])

            ax.text(
                x,
                5,  # slightly below top edge
                label,
                color=colour,
                fontsize=8,
                ha="center",
                va="top",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none")
            )

        return fig

    def create_detection_video(
            self,
            detections_df=None,
            out_path: str = "detections.mp4",
            metric_col: str = "impact_score",
            box_width: int = 5,
            fps: int = 2,
            tmp_downscale: bool = False
    ):
        """
        Render every pre-processed frame with coloured bounding boxes and
        assemble an MP4.  Frames that have *no* detections are shown as
        greyscale only.

        Parameters
        ----------
        detections_df : pd.DataFrame or None
            Master table of detections. If None, uses self.annotated_data.
        out_path : str
            Destination filename (extension decides container/codec).
        metric_col, box_width, fps, tmp_downscale
            Same meaning as before.

        Returns
        -------
        pathlib.Path
            Absolute path to the written video file.
        """

        if detections_df is None:
            if hasattr(self, "annotated_data") and self.annotated_data is not None:
                detections_df = self.annotated_data
            else:
                raise ValueError(
                    "detections_df was None and self.annotated_data is missing."
                )

        max_h, max_w = 0, 0
        for _, img in self.preprocessed_data:
            h, w = img.shape
            if tmp_downscale:
                h //= 2
                w //= 2
            max_h, max_w = max(max_h, h), max(max_w, w)

        frames_np = []

        for frame_name, img in self.preprocessed_data:

            try:
                fig = self.draw_detections_on_frame(
                    frame_name=frame_name,
                    detections_df=detections_df,
                    metric_col=metric_col,
                    box_width=box_width
                )
            except ValueError:  # "No detections for frame …"
                disp = cv2.normalize(
                    img, None, alpha=0, beta=255,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                fig, ax = plt.subplots(
                    figsize=(disp.shape[1] / 100, disp.shape[0] / 100), dpi=100
                )
                ax.imshow(disp, cmap="gray", vmin=0, vmax=255, aspect="auto")
                ax.set_axis_off()
                fig.tight_layout(pad=0)

            fig.canvas.draw()
            w_fig, h_fig = fig.canvas.get_width_height()
            rgba = np.asarray(fig.canvas.buffer_rgba()).reshape((h_fig, w_fig, 4))
            rgb = rgba[:, :, :3].copy()

            plt.close(fig)

            if tmp_downscale:
                rgb = rgb[::2, ::2, :]

            pad_h, pad_w = max_h - rgb.shape[0], max_w - rgb.shape[1]
            if pad_h or pad_w:
                rgb = np.pad(
                    rgb,
                    pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0
                )

            frames_np.append(rgb)

        out_path = pathlib.Path(out_path).expanduser().resolve()
        clip = ImageSequenceClip(frames_np, fps=fps)
        clip.write_videofile(
            out_path.as_posix(),
            codec="libx264",
            bitrate="4000k",
            preset="medium",
            audio=False,
            logger=None
        )

        print(f"Video saved → {out_path}")
        return out_path
