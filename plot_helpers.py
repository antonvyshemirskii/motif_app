import copy
import pandas as pd
import numpy as np
import EntropyHub as EH
import tqdm
import umap
import warnings
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.neighbors import NearestNeighbors
from DTW_wrapper import DTW
from constants import (
    TARGET_COL_1,
    TARGET_COL_2,
    COLOR_COL,
    SOURCE_TARGET_COL_1,
    SOURCE_TARGET_COL_2,
    SOURCE_DIR,
)

warnings.simplefilter("ignore")


def create_entropy():
    # TODO: find out how to create real entropy
    entropies = pd.read_csv(os.path.join(SOURCE_DIR, "entropies_all.csv"), index_col=0)
    entropies = entropies.replace([-np.inf, np.inf], np.nan).dropna()
    return entropies


def get_top_entropy_samples(entropies, top_samples=5000):
    """
    Get the top N samples from the input entropies DataFrame.

    Parameters:
    - entropies: DataFrame containing entropy values.
    - top_samples: Number of top samples to select (default is 5000).

    Returns:
    - DataFrame with the top N samples.
    """
    entropies_sample = copy.deepcopy(entropies.iloc[:top_samples, :])
    return entropies_sample


def apply_umap(data, n_components=None):
    """
    Apply UMAP (Uniform Manifold Approximation and Projection) to the input data.

    Parameters:
    - data: Input data (DataFrame or array-like).
    - n_components: Number of dimensions in the output space (default is None).

    Returns:
    - UMAP-transformed data.
    """
    if n_components is None:
        umap_model = umap.UMAP()
    else:
        umap_model = umap.UMAP(n_components=n_components)

    dimred = umap_model.fit_transform(data)
    return dimred


def create_output(source_file):
    # dtw_source = DTW(zeroing=True, stitched_data=True)
    # _ = dtw_source.load_h5(source_file)
    # (
    #     source_resampled,
    #     select_files,
    #     select_columns,
    #     resample_string,
    # ) = dtw_source.reconstruct_master_df()
    # num_features, masking_factor = dtw_source.get_dtw_params()

    # entropies = create_entropy()
    # entropies_sample = get_top_entropy_samples(entropies)

    # dimred = apply_umap(entropies_sample.values)
    # dimred_color = apply_umap(entropies_sample.values, n_components=1)

    # entropies_sample[COLOR_COL] = dimred_color
    # entropies_sample[TARGET_COL_1] = dimred[:, 0]
    # entropies_sample[TARGET_COL_2] = dimred[:, 1]

    # plot_data = entropies_sample[[TARGET_COL_1, TARGET_COL_2, COLOR_COL]]
    # source_data = source_resampled[
    #     [*select_columns, SOURCE_TARGET_COL_1, SOURCE_TARGET_COL_2]
    # ]
    # plot_data.to_csv(os.path.join(SOURCE_DIR, "plot_data.csv"))
    # source_data.to_csv(os.path.join(SOURCE_DIR, "source_data.csv"))
    # placeholders
    plot_data = pd.read_csv(os.path.join(SOURCE_DIR, "plot_data.csv"), index_col=0)
    source_data = pd.read_csv(os.path.join(SOURCE_DIR, "source_data.csv"), index_col=0)
    num_features = 63
    return plot_data, source_data, num_features
