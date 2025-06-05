import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.edge_detect import EdgeDetector
from src.reading import Reader
from src.readjson import ReaderJson


def compute_confusion_elements(y_true, y_pred, tolerance=0):
    """
    Computes TP, FP, TN, FN between binary ground truth and prediction with tolerance.
    A prediction is considered a true positive if there is any true positive within the tolerance window.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    matched_true = set()
    matched_pred = set()

    for i, pred_val in enumerate(y_pred):
        if pred_val == 1:
            start = max(0, i - tolerance)
            end = min(len(y_true), i + tolerance + 1)
            for j in range(start, end):
                if y_true[j] == 1 and j not in matched_true:
                    matched_true.add(j)
                    matched_pred.add(i)
                    break

    tp = len(matched_pred)
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_true) - len(matched_true)
    tn = len(y_true) - tp - fp - fn

    return tp, fp, tn, fn


class Evaluator:
    def __init__(self, relative_path):
        current_dir = os.getcwd()
        json_path = os.path.join(current_dir, f'{relative_path}.json')
        h5_path = os.path.join(current_dir, f'{relative_path}.h5')
        self.readerjson = ReaderJson(json_path)
        self.reader = Reader(h5_path)
        self.df_json = self.readerjson.return_data()
        self.df_h5 = self.reader.return_data()

    def predict_histogram(self, threshold_hist=1.4):
        # Predict histogram edges for all images
        return [
            EdgeDetector(self.df_h5.loc[idx]['matrix']).detect_edges_histogram(threshold_value=threshold_hist)
            for idx in self.df_json['image_index']
        ]

    def predict_radon(self, threshold_radon=1.4):
        # Predict radon edges for all images
        return [
            EdgeDetector(self.df_h5.loc[idx]['matrix']).detect_edges_radon(threshold=threshold_radon)
            for idx in self.df_json['image_index']
        ]

    def predict_canny(self, percentile_canny=98):
        # Predict canny edges for all images
        return [
            EdgeDetector(self.df_h5.loc[idx]['matrix']).detect_edges_canny(percentile=percentile_canny, roc=True)
            for idx in self.df_json['image_index']
        ]

    def create_histogram_predictions(self, threshold_hist=1.4):
        # Add histogram predictions to dataframe
        self.df_json['peaks_hist'] = self.predict_histogram(threshold_hist=threshold_hist)
        self.df = self.df_json
        return self.df

    def create_radon_predictions(self, threshold_radon=1.4):
        # Add radon predictions to dataframe
        self.df_json['peaks_radon'] = self.predict_radon(threshold_radon=threshold_radon)
        self.df = self.df_json
        return self.df

    def create_canny_predictions(self, percentile_canny=98):
        # Add canny predictions to dataframe
        self.df_json['peaks_canny'] = self.predict_canny(percentile_canny=percentile_canny)
        self.df = self.df_json
        return self.df

    def create_binary_mask(self, method):
        '''
        Create binary masks for ground truth and a given prediction method (either "canny" or "hist")
        Returns a dataframe with columns: image_idx, binary_true, binary_pred, num_true, num_pred
        '''
        matrix_len = self.df_h5.iloc[0]['matrix'].shape[1]
        data = []
        for image_idx, row in self.df.iterrows():
            binary_mask_true = np.zeros(matrix_len)
            binary_mask_pred = np.zeros(matrix_len)
            true = row['true']
            if method == 'canny':
                pred = row['peaks_canny']
            elif method == 'hist':
                pred = row['peaks_hist']
            elif method == 'radon':
                pred = row['peaks_radon']
            else:
                raise ValueError("Method must be 'canny' or 'hist'")
            binary_mask_true[true] = 1
            binary_mask_pred[pred] = 1
            data.append({
                'image_idx': image_idx,
                'binary_true': binary_mask_true,
                'binary_pred': binary_mask_pred,
                'num_true': sum(binary_mask_true),
                'num_pred': sum(binary_mask_pred)
            })
        df_binary = pd.DataFrame(data)
        return df_binary

    def metrics(self, df_binary, tolerance=5):
        # Compute metrics for a given binary mask dataframe, that is the metrics for the whole sequence
        tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0
        for idx, row in df_binary.iterrows():
            true = row['binary_true']
            pred = row['binary_pred']
            tp, fp, tn, fn = compute_confusion_elements(true, pred, tolerance=tolerance)
            tp_total += tp
            fp_total += fp
            tn_total += tn
            fn_total += fn
        num_true = sum(df_binary['num_true'])
        num_pred = sum(df_binary['num_pred'])
        return {
            'TP': tp_total,
            'FP': fp_total,
            'TN': tn_total,
            'FN': fn_total,
            'total_true': num_true,
            'total_pred': num_pred
        }

    def _plot_confusion_matrix(self, cm, title):
        """
        Helper to plot a confusion matrix.
        cm: 2x2 array [[TN, FP], [FN, TP]]
        title: string for plot title
        """
        plt.figure(figsize=(3.8, 2.8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar(shrink=0.9)
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Normal', 'Defect'])
        plt.yticks(tick_marks, ['Normal', 'Defect'])
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f'{title}.pdf')
        plt.show()

    def roc(self, method, hist_values=None, canny_values=None, plot_confusion=False):
        '''
        Compute ROC curve data for the specified method.
        method: 'canny' or 'hist'
        '''
        if hist_values is None:
            hist_values = list(np.arange(0.9, 1.34, 0.04)) + list(np.arange(1.4, 2.1, 0.1))
        if canny_values is None:
            canny_values = np.linspace(0, 100, 10)
        results = []
        if method == 'canny':
            for percentile_canny in tqdm(canny_values, desc='Canny ROC'):
                self.create_canny_predictions(percentile_canny=percentile_canny)
                self.df = self.df_json
                df_binary = self.create_binary_mask('canny')
                metrics = self.metrics(df_binary)
                tp, fp, tn, fn = metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                results.append({
                    'percentile_canny': percentile_canny,
                    'TPR': tpr,
                    'FPR': fpr
                })
            df_results = pd.DataFrame(results)
            # Compute Youden's J statistic and print best percentile_canny
            df_results['Youden_J'] = df_results['TPR'] - df_results['FPR']
            best_row = df_results.loc[df_results['Youden_J'].idxmax()]
            best_threshold = best_row['percentile_canny']
            print(f"Best percentile_canny for Youden's J: {best_threshold}")
            # Compute distance to (0,1)
            df_results['distance'] = np.sqrt(df_results['FPR'] ** 2 + (1 - df_results['TPR']) ** 2)
            # Grab the row with smallest distance
            best0_row = df_results.loc[df_results['distance'].idxmin()]
            best_balanced_threshold = best0_row['percentile_canny']
            print(f"Best percentile by 0-1 corner method: {best_balanced_threshold}")

            # print(df_results)

            if plot_confusion:
                # Generate predictions at the chosen threshold
                self.create_canny_predictions(percentile_canny=best_balanced_threshold)
                self.df = self.df_json
                df_binary = self.create_binary_mask('canny')
                metrics_dict = self.metrics(df_binary)
                TP = metrics_dict['TP']
                FP = metrics_dict['FP']
                TN = metrics_dict['TN']
                FN = metrics_dict['FN']
                cm = np.array([[TN, FP], [FN, TP]])
                self._plot_confusion_matrix(cm, f'CM (Canny). Percentile {best_balanced_threshold:.1f}')
            return df_results
        elif method == 'hist':
            for threshold_hist in tqdm(hist_values, desc='Hist ROC'):
                self.create_histogram_predictions(threshold_hist=threshold_hist)
                self.df = self.df_json
                df_binary = self.create_binary_mask('hist')
                metrics = self.metrics(df_binary)
                tp, fp, tn, fn = metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                results.append({
                    'threshold_hist': threshold_hist,
                    'TPR': tpr,
                    'FPR': fpr
                })
            df_results = pd.DataFrame(results)
            # Compute Youden's J statistic and print best threshold_hist
            df_results['Youden_J'] = df_results['TPR'] - df_results['FPR']
            best_row = df_results.loc[df_results['Youden_J'].idxmax()]
            best_threshold = best_row['threshold_hist']
            print(f"Best threshold_hist for Youden's J: {best_threshold}")
            # Compute distance to (0,1)
            df_results['distance'] = np.sqrt(df_results['FPR'] ** 2 + (1 - df_results['TPR']) ** 2)
            # Grab the row with smallest distance
            best0_row = df_results.loc[df_results['distance'].idxmin()]
            best_balanced_threshold = best0_row['threshold_hist']
            print(f"Best threshold by 0-1 corner method: {best_balanced_threshold}")
            # print(df_results)

            if plot_confusion:
                # Generate predictions at the chosen threshold
                self.create_histogram_predictions(threshold_hist=best_balanced_threshold)
                self.df = self.df_json
                df_binary = self.create_binary_mask('hist')
                metrics_dict = self.metrics(df_binary)
                TP = metrics_dict['TP']
                FP = metrics_dict['FP']
                TN = metrics_dict['TN']
                FN = metrics_dict['FN']
                cm = np.array([[TN, FP], [FN, TP]])
                self._plot_confusion_matrix(cm, f'CM (Histogram). Threshold value {best_threshold:.2f}')
            return df_results

        elif method == 'radon':
            for threshold_radon in tqdm(hist_values, desc="Radon ROC"):
                self.create_radon_predictions(threshold_radon=threshold_radon)
                self.df = self.df_json
                df_binary = self.create_binary_mask('radon')
                metrics = self.metrics(df_binary)
                tp, fp, tn, fn = metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                results.append({
                    'threshold_radon': threshold_radon,
                    'TPR': tpr,
                    'FPR': fpr
                })
            df_results = pd.DataFrame(results)
            # Compute Youden's J statistic and print best threshold_radon
            df_results['Youden_J'] = df_results['TPR'] - df_results['FPR']
            best_row = df_results.loc[df_results['Youden_J'].idxmax()]
            best_threshold = best_row['threshold_radon']
            print(f"Best threshold_radon for Youden's J: {best_threshold}")
            # Compute distance to (0,1)
            df_results['distance'] = np.sqrt(df_results['FPR'] ** 2 + (1 - df_results['TPR']) ** 2)
            # Grab the row with smallest distance
            best0_row = df_results.loc[df_results['distance'].idxmin()]
            best_balanced_threshold = best0_row['threshold_radon']
            print(f"Best threshold by 0-1 corner method: {best_balanced_threshold}")
            # print(df_results)
            if plot_confusion:
                # Generate predictions at the chosen threshold
                self.create_radon_predictions(threshold_radon=best_balanced_threshold)
                self.df = self.df_json
                df_binary = self.create_binary_mask('radon')
                metrics_dict = self.metrics(df_binary)
                TP = metrics_dict['TP']
                FP = metrics_dict['FP']
                TN = metrics_dict['TN']
                FN = metrics_dict['FN']
                cm = np.array([[TN, FP], [FN, TP]])
                self._plot_confusion_matrix(cm, f'CM (Radon). Threshold value {best_threshold:.2f}')
            return df_results

    def plot_roc(self, hist_values=None, canny_values=None, plot_confusion=False):
        # ROC curves
        roc_hist = self.roc('hist', hist_values=hist_values, canny_values=canny_values, plot_confusion=plot_confusion)
        roc_canny = self.roc('canny', hist_values=hist_values, canny_values=canny_values, plot_confusion=plot_confusion)
        roc_radon = self.roc('radon', hist_values=hist_values, canny_values=canny_values, plot_confusion=plot_confusion)
        # Plot ROC curves
        plt.figure(figsize=(4.5, 3.7))
        plt.plot(roc_hist['FPR'], roc_hist['TPR'], marker='x', label='Histogram')
        plt.plot(roc_canny['FPR'], roc_canny['TPR'], marker='o', label='Canny')
        plt.plot(roc_radon['FPR'], roc_radon['TPR'], marker='*', label='Radon')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Histogram vs Canny vs Radon')
        plt.legend()
        # plt.grid(True)
        plt.savefig('ROC.pdf')
        plt.show()

        # # Create figure with two subplots: main ROC and zoomed-in
        # fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(8, 3))

        # # Main ROC plot
        # ax_main.plot(roc_hist['FPR'], roc_hist['TPR'], marker='x', label='Histogram')
        # ax_main.plot(roc_canny['FPR'], roc_canny['TPR'], marker='o', label='Canny')
        # ax_main.plot(roc_radon['FPR'], roc_radon['TPR'], marker='*', label='Radon')
        # ax_main.plot([0, 1], [0, 1], 'k--')
        # ax_main.set_xlabel('False Positive Rate')
        # ax_main.set_ylabel('True Positive Rate')
        # ax_main.set_title('ROC Curve - Histogram vs Canny vs Radon')
        # ax_main.legend()
        # #ax_main.grid(True)

        # # Zoomed-in ROC (e.g., FPR < 0.2, TPR > 0.8)
        # ax_zoom.plot(roc_hist['FPR'], roc_hist['TPR'], marker='x')
        # ax_zoom.plot(roc_canny['FPR'], roc_canny['TPR'], marker='o')
        # ax_zoom.plot(roc_radon['FPR'], roc_radon['TPR'], marker='*')
        # ax_zoom.set_xlim(0, 0.2)
        # ax_zoom.set_ylim(0.8, 1.0)
        # ax_zoom.set_title('Zoomed In (FPR < 0.2, TPR > 0.8)')
        # ax_zoom.set_xlabel('False Positive Rate')
        # ax_zoom.set_ylabel('True Positive Rate')
        # #ax_zoom.grid(True)

        # plt.tight_layout()
        # plt.savefig('ROC_with_zoom.pdf')
        # plt.show()


if __name__ == '__main__':
    relative_path = 'raw_data/Données_CN_V1/SCORPIO_LWIR/SCORPIO-LW_C1_REF_N1'
    # relative_path = 'raw_data/Données_CN_V1/SEEGNUS/SEEGNUS_C1_REF_N1'
    evaluator = Evaluator(relative_path)

    # Determine thresholds to iterate over
    hist_values = list(np.arange(0.9, 1.34, 0.04)) + list(np.arange(1.4, 2.2, 0.1))
    canny_values = np.linspace(0, 100, 10)

    evaluator.plot_roc(hist_values=hist_values, canny_values=canny_values, plot_confusion=True)
