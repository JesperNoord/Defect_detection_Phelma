import numpy as np
from src.evaluate_models import Evaluator


if __name__ == '__main__':
    relative_path = 'raw_data/Données_CN_V1/SCORPIO_LWIR/SCORPIO-LW_C1_REF_N1'
    #relative_path = 'raw_data/Données_CN_V1/SEEGNUS/SEEGNUS_C1_REF_N1'
    evaluator = Evaluator(relative_path)

    # Determine thresholds to iterate over
    hist_values = list(np.arange(0.9, 1.34, 0.04)) + list(np.arange(1.4, 2.2, 0.1))
    canny_values = np.linspace(0, 100, 10)

    evaluator.plot_roc(hist_values=hist_values, canny_values=canny_values, plot_confusion=True)