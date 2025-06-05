from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.edge_detect as edge_detect


def _detect_edges_for_image(name, image, settings):
    """
    Helper that will run in a worker process.
    Returns (name, edges) so we can rebuild detections in the main process.
    """
    detector = edge_detect.EdgeDetector(image)
    edges = detector.detect_defects(
        settings['detection_method'],
        **settings
    )
    return name, edges


class EdgeProcessor:
    def __init__(self, images, settings):
        # store settings & images
        self.settings = settings
        self.images = images

        # prepare the structure to hold names and edges
        self.detections = {
            'name': [],
            'edges': [],
        }
        self.PARAMETER_MIN_FRAGMENTED_DISTANCE = settings['fragmented_min_distance']  # must be odd
        self.PARAMETER_FRAGMENTED_THRESHOLD_MULTIPLITER = .9
        self.PARAMETER_FRAGMENTED_THRESHOLD_HYSTERESIS_MULTIPLITER = 0.3

        # Extract the method name once (no need to look it up inside each worker repeatedly)
        detect_defect_method = settings['detection_method']

        # Build a small payload to send to each worker:
        #   (name, image, settings).  We assume `settings` is picklable.
        tasks = [
            (name, image, settings)
            for name, image in self.images
        ]

        # Use a ProcessPoolExecutor to parallelize across CPU cores.
        # You could also choose ThreadPoolExecutor if `detect_defects` releases the GIL,
        # but in most image‐processing cases multiprocessing gives better CPU utilization.
        with ProcessPoolExecutor() as executor:
            # Submit all tasks at once:
            future_to_pair = {
                executor.submit(_detect_edges_for_image, name, image, settings): (name, image)
                for name, image, settings in tasks
            }

            # As each worker finishes, append its result:
            for future in as_completed(future_to_pair):
                try:
                    name, edges = future.result()
                except Exception as exc:
                    # If a worker crashed, you can decide how to handle it.
                    # For now, we'll just skip.
                    print(f"Image‐processing for {future_to_pair[future][0]} raised {exc!r}")
                    continue

                self.detections['name'].append(name)
                self.detections['edges'].append(edges)

        # At this point, self.detections is filled in the same way the original loop would have.
        self.input_df = pd.DataFrame(self.detections)
        self.df = self.unpack_edges()

    def unpack_edges(self):
        out_data = {
            'frame': [],
            'position': []
        }

        for index, data in self.input_df.iterrows():
            if not data['edges'].tolist():
                continue

            for column in data['edges']:
                out_data['frame'].append(data['name'])
                out_data['position'].append(column)
        return pd.DataFrame(out_data)

    def get_column(self, image_name, column, show=False):

        ix = self.images.find_index_by_name(image_name)
        image = self.images[ix][1]

        if show:
            plt.figure(figsize=(10, .5))
            plt.imshow(image[:, column].reshape(-1, 1).transpose(), cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return image[:, column].reshape(-1, 1).transpose()

    def detect_fragmented(self, im=None, col=None):
        if not im and not col:
            self.df['type'] = self.df.apply(self._fragmented_parser, axis=1)
            # self._fragmented_growing('Image 0001', 0, False)
            return self.df

        else:
            print('detecting defect in {} col within {} img'.format(col, im))
            self._fragmented_growing(im, col, True)
        pass

    def _get_normal_stats(self, im, col, skip='none'):
        """
        Calculates and retrieves the mean (mu) and standard deviation (sigma)
        for a specified column in the image requested. This function checks
        to ensure the column index is within valid bounds and that the
        specified column is not defective. In cases where the column is
        defective or out of bounds, the function does not perform further
        processing and returns None values.

        :param im: The name of the image for which statistics are to be
            calculated. Must be a valid image name that is present in the
            database.
        :type im: str
        :param col: The column index within the specified image for which
            statistics will be calculated. Must be a non-negative integer
            and within the column bounds of the image.
        :type col: int
        :param skip:
        'none' - returns none
        'left' - go to the left until a valid column is found
        'right' - go to the right until a valid column is found

        :type skip: str, optional
        :return: A tuple containing:
            - mu (float): The mean of the specified column if the column is
              valid and non-defective.
            - sigma (float): The standard deviation of the specified column
              if the column is valid and non-defective.
            Returns (None, None) if the column is defective or out of bounds.
        :rtype: tuple[float | None, float | None]
        """
        # start with ifs

        im_ind = self.images.find_index_by_name(im)
        im_shape = self.images[im_ind][1].shape[1]

        if col > im_shape - 1 or col < 0:
            return None, None

        mask = (self.df['frame'] == im) & (self.df['position'] == col)
        if mask.any():
            if skip == 'left':
                mu, sigma = self._get_normal_stats(im, col - 1, skip='left')
                return mu, sigma
            elif skip == 'right':
                mu, sigma = self._get_normal_stats(im, col + 1, skip='right')
                return mu, sigma
            else:
                return None, None

        edge = self.get_column(im, col)
        mu = edge.mean()
        sigma = edge.std()
        return mu, sigma

    def _fragmented_growing(self, im, col, show=False, check=True):

        edge = self.get_column(im, col)
        original_edge = edge.copy()
        sz = edge.shape[1]
        img_x_sz = self.images[0][1].shape[1]
        if show:
            plt.plot(edge[0, :])

        edge = cv2.GaussianBlur(edge, (43, 1), 15)
        if show:
            plt.plot(edge[0, :])

        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((1, self.PARAMETER_MIN_FRAGMENTED_DISTANCE)))
        if show:
            plt.plot(edge[0, :])

        ## calculate mu, sigma
        # from left and right cols

        cols_check = np.array([col - 2, col - 1, col + 1, col + 2])
        if col < 2:
            cols_check += 3
        elif col > img_x_sz - 2:
            cols_check -= 3

        mu_mi = []

        sigma_mi = []
        for col_check in cols_check:
            side = 'left' if col_check - col < 0 else 'right'
            mu, sigma = self._get_normal_stats(im, col_check, side)
            mu_mi.append(mu)
            sigma_mi.append(sigma)
        mu_mi = np.array(mu)
        sigma_mi = np.array(sigma)
        mu_mi = mu_mi[mu_mi != None]
        sigma_mi = sigma_mi[sigma_mi != None]

        mu = np.mean(mu_mi)
        sigma = np.mean(sigma_mi)

        threshold_center = mu + self.PARAMETER_FRAGMENTED_THRESHOLD_MULTIPLITER * sigma
        threshold_bot = threshold_center - (sigma * self.PARAMETER_FRAGMENTED_THRESHOLD_HYSTERESIS_MULTIPLITER)
        threshold_top = threshold_center + (sigma * self.PARAMETER_FRAGMENTED_THRESHOLD_HYSTERESIS_MULTIPLITER)

        boolean_edge = np.zeros_like(edge)
        state = False

        for i, pix in enumerate(edge[0, :]):
            if not state and pix > threshold_top:
                state = True
            elif state and pix < threshold_bot:
                state = False
            boolean_edge[0, i] = state

        if np.sum(boolean_edge[0, :]) == 0 and check == True and col >= 1:
            if show:
                print('probably wrong edge, checking on left')

            boolean_edge = self._fragmented_growing(im, col - 1, show=False, check=False)

        if np.sum(boolean_edge[0, :]) == 0 and check == True and col < img_x_sz:
            if show:
                print('probably wrong edge, checking on right')
            boolean_edge = self._fragmented_growing(im, col + 1, show=False, check=False)

        if np.sum(boolean_edge[0, :]) == 0 and check == True:
            # print('Warning: edge in column {} not found in {}'.format(col, im))
            pass

        if len(boolean_edge[0, :]) * .9 < sum(boolean_edge[0, :]):
            output = 'not fragmented'
        elif sum(boolean_edge[0, :]) == 0:
            output = 'not fragmented'
        else:
            output = 'fragmented'

        if check and show:
            print('\n\nstats for {} column in {}'.format(col, im))
            print('length of column: {}'.format(len(boolean_edge[0, :])))
            print('sum of edges: {}'.format(np.sum(boolean_edge[0, :])))

            if len(boolean_edge[0, :]) == sum(boolean_edge[0, :]):
                print('not fragmented')
            else:
                print('fragmented')

            print('\n\n')

        if show:
            print(boolean_edge)
            plt.plot(np.arange(len(edge[0, :])), np.ones(len(edge[0, :])) * threshold_center, 'r')
            plt.plot(np.arange(len(edge[0, :])), np.ones(len(edge[0, :])) * threshold_bot, 'g--')
            plt.plot(np.arange(len(edge[0, :])), np.ones(len(edge[0, :])) * threshold_top, 'g--')

        if show:
            plt.subplots(3, 1, figsize=(10, .5))

            plt.subplot(3, 1, 1)
            plt.imshow(original_edge, cmap='gray')
            plt.axis('off')

            plt.subplot(3, 1, 2)
            plt.imshow(edge, cmap='gray')
            plt.axis('off')

            plt.subplot(3, 1, 3)
            plt.imshow(boolean_edge, cmap='rainbow', vmin=0, vmax=1)
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        if check:
            return output
        else:
            return boolean_edge

    def detect_blinking_or_noisy(
            self,
            detections: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Classify columns (NOISY / BLINKING) and give both a summary table and a
        per-detection table with the column information merged in.
        """

        # ───────────────────────── parameters ──────────────────────────
        PARAMETER_TOLERANCE_PX = 1  # ±px for clustering
        PARAMETER_NOISY_THRESHOLD_PCT = 0.95  # ≥ → NOISY
        # ----------------------------------------------------------------
        total_frames = len(self.images)
        # 0) safety copy
        detections = detections.copy()

        # 1) cluster positions → col_id, mean_x
        unique_pos = sorted(detections['position'].unique())
        if not unique_pos:
            raise ValueError("No detections supplied")

        clusters, cur = [], [unique_pos[0]]
        for x in unique_pos[1:]:
            if x - cur[-1] <= PARAMETER_TOLERANCE_PX:
                cur.append(x)
            else:
                clusters.append(cur)
                cur = [x]
        clusters.append(cur)

        pos2cid, cid2mean = {}, {}
        for cid, grp in enumerate(clusters):
            for p in grp:
                pos2cid[p] = cid
            cid2mean[cid] = int(np.mean(grp))

        detections["col_id"] = detections["position"].map(pos2cid)

        # 2) presence matrix
        frames = sorted(detections["frame"].unique())
        col_ids = sorted(detections["col_id"].unique())
        presence = pd.DataFrame(0, index=col_ids, columns=frames)
        for _, r in detections.iterrows():
            presence.at[r["col_id"], r["frame"]] = 1

        # 3) column-level ‘ever fragmented?’
        col_has_fragmented = (
            detections.assign(is_frag=detections["type"].eq("fragmented"))
            .groupby("col_id")["is_frag"]
            .any()
        )

        # 4) build summary
        summary_rows = []
        for cid, row in presence.iterrows():
            present_pct = row.sum() / total_frames
            status = (
                "NOISY" if present_pct >= PARAMETER_NOISY_THRESHOLD_PCT
                else "BLINKING"
            )
            summary_rows.append(
                dict(
                    col_id=cid,
                    mean_x=cid2mean[cid],
                    presence_pct=round(present_pct, 3),
                    status=status,
                    col_has_fragmented=bool(col_has_fragmented.get(cid, False)),
                    frames_list=",".join([f for f, v in zip(presence.columns, row.values) if v]),
                )
            )

        summary_df = pd.DataFrame(summary_rows)

        # 5) merge summary info back to every detection row
        annotated_df = detections.merge(
            summary_df[["col_id", "status", "presence_pct", "col_has_fragmented"]],
            on="col_id",
            how="left"
        )

        return summary_df, annotated_df

    def append_intensity_metrics(
            self,
            detections_df: pd.DataFrame,
            window_width: int = 10
    ) -> pd.DataFrame:
        """
        Enrich *detections_df* with per-detection intensity metrics.

        Parameters
        ----------
        detections_df : pd.DataFrame
            Must contain at least:
                • frame     – frame name exactly as in self.images
                • position  – x coordinate (int)
        window_width : int, default 10
            Horizontal width (pixels) of the analysis strip around each defect.

        Returns
        -------
        pd.DataFrame
            Copy of *detections_df* with columns appended:
                mean_intensity, peak_intensity, background_intensity,
                contrast_ratio, vertical_coverage, impact_score, severity
        """

        # ──────────── PARAMETERS (visible, easy to tweak) ────────────
        PARAMETER_WINDOW_WIDTH = window_width
        PARAMETER_BG_EXCLUSION_WIDTH = window_width  # ±px masked out
        PARAMETER_THRESHOLD_FACTOR = 1.10  # 10 % > background
        # -------------------------------------------------------------

        df = detections_df.copy()

        # Pre-allocate the new columns with NaN
        new_cols = [
            "mean_intensity", "peak_intensity", "background_intensity",
            "contrast_ratio", "vertical_coverage", "impact_score", "severity"
        ]
        for col in new_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Work frame-by-frame to minimise repeated background work
        for frame_name in df["frame"].unique():
            row_mask = df["frame"] == frame_name
            if not row_mask.any():
                continue  # should not happen, but be safe

            # ── fetch image for this frame
            try:
                n = self.images.find_index_by_name(frame_name)
            except AttributeError:
                raise RuntimeError(
                    "self.images must implement find_index_by_name(frame_name)"
                )
            _, image = self.images[n]  # (name, image) tuple
            W = image.shape[1]

            # ── background estimation : mask out ±exclusion around each defect
            bg_mask = np.ones(W, dtype=bool)
            for pos in df.loc[row_mask, "position"]:
                s = max(0, pos - PARAMETER_BG_EXCLUSION_WIDTH)
                e = min(W, pos + PARAMETER_BG_EXCLUSION_WIDTH + 1)
                bg_mask[s:e] = False
            background_intensity = float(np.mean(image[:, bg_mask]))

            # ── per-detection metrics for this frame
            contrast_vals = []
            metric_cache = {}  # keyed by DataFrame index

            half_win = PARAMETER_WINDOW_WIDTH // 2
            for idx, pos in zip(df.loc[row_mask].index,
                                df.loc[row_mask, "position"]):
                c0 = max(0, pos - half_win)
                c1 = min(W, pos + half_win + 1)
                region = image[:, c0:c1]

                mean_int = float(np.mean(region))
                peak_int = float(np.max(region))
                contrast = peak_int / (background_intensity + 1e-6)
                thresh = background_intensity * PARAMETER_THRESHOLD_FACTOR
                vert_cov = np.count_nonzero(region > thresh) / region.shape[0]
                impact = contrast * vert_cov

                metric_cache[idx] = (
                    mean_int, peak_int, contrast, vert_cov, impact
                )
                contrast_vals.append(contrast)

            # ── severity thresholds (quartiles of *this frame* only)
            if len(contrast_vals) >= 4:
                q1, q2, q3 = np.percentile(contrast_vals, [25, 50, 75])
            else:
                q1 = q2 = q3 = np.median(contrast_vals)

            def severity_label(r):
                if r < q1:
                    return "Minor"
                elif r < q2:
                    return "Moderate"
                elif r < q3:
                    return "Severe"
                else:
                    return "Critical"

            # ── write results back into df
            for idx, (mean_int, peak_int, contrast, vcov, impact) in metric_cache.items():
                df.at[idx, "mean_intensity"] = mean_int
                df.at[idx, "peak_intensity"] = peak_int
                df.at[idx, "background_intensity"] = background_intensity
                df.at[idx, "contrast_ratio"] = contrast
                df.at[idx, "vertical_coverage"] = vcov
                df.at[idx, "impact_score"] = impact
                df.at[idx, "severity"] = severity_label(contrast)

        return df

    def _fragmented_parser(self, row):
        return self._fragmented_growing(row['frame'], row['position'], show=False)

    def plot_histogram(self, image_name):
        ix = self.images.find_index_by_name(image_name)
        image = self.images[ix][1]
        hist = cv2.calcHist(image, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(hist)


if __name__ == '__main__':
    a = EdgeProcessor(
        '/home/maciejka/Desktop/school/S8/kurwa/raw_data/Données_CN_V3/SCORPIO_LWIR/SCORPIO-LW_C3_RAW_N1.h5')
    # a.images.show_image(25)`
    print(a.df)
    a.detect_fragmented()

    print(a.df)
