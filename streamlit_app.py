import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from plot_helpers import create_output
from time import sleep
from streamlit_plotly_events import plotly_events
from sklearn.neighbors import NearestNeighbors
from constants import (
    TARGET_COL_1,
    TARGET_COL_2,
    COLOR_COL,
    SOURCE_TARGET_COL_1,
    SOURCE_TARGET_COL_2,
    SOURCE_DIR,
)

session_state = st.session_state
if not hasattr(session_state, "latest_chosen_point"):
    session_state.latest_chosen_point = []

if not hasattr(session_state, "current_figure"):
    session_state.current_figure = None

if not hasattr(session_state, "source_path"):
    session_state.source_path = ""

if not hasattr(session_state, "plot_data"):
    session_state.plot_data = None

if not hasattr(session_state, "source_data"):
    session_state.source_resampled = None

if not hasattr(session_state, "num_features"):
    session_state.num_features = None

if not hasattr(session_state, "plotting"):
    session_state.plotting = False

if not hasattr(session_state, "n_neighbors"):
    session_state.n_neighbors = 0


def find_point_index(x_col, y_col, x_col_value, y_col_value):
    """
    Find the index of a point in the DataFrame based on known x and y values.

    Parameters:
    - x_col: Column name or index for the x-axis.
    - y_col: Column name or index for the y-axis.
    - x_col_value: Value of x_col for the point.
    - y_col_value: Value of y_col for the point.

    Returns:
    - Index of the point in the DataFrame.
    """
    point_index = session_state.plot_data[
        (session_state.plot_data[x_col] == x_col_value)
        & (session_state.plot_data[y_col] == y_col_value)
    ].index
    if not point_index.empty:
        return point_index[0]
    else:
        return None


def find_point_sequence(x_col, y_col, x_col_value, y_col_value):
    """
    Find the sequence number of a point in the DataFrame based on known x and y values.

    Parameters:
    - x_col: Column name or index for the x-axis.
    - y_col: Column name or index for the y-axis.
    - x_col_value: Value of x_col for the point.
    - y_col_value: Value of y_col for the point.

    Returns:
    - Sequence number of the point in the DataFrame.
    """
    # Reset index to default integer-based index
    data_reset = session_state.plot_data.reset_index(drop=True)

    # Find the index of the point
    point_index = data_reset[
        (data_reset[x_col] == x_col_value) & (data_reset[y_col] == y_col_value)
    ].index

    if not point_index.empty:
        return point_index[0]
    else:
        return None


def update_umap_scatter(
    x_col,
    y_col,
    color_col,
    size=0.5,
    x_known=None,
    y_known=None,
    larger_point_size=4,
    marker_col="marker_size",
):
    session_state.plot_data[marker_col] = size

    # Update the existing figure
    session_state.current_figure.update_traces(
        x=session_state.plot_data[x_col],
        y=session_state.plot_data[y_col],
        marker=dict(size=session_state.plot_data[marker_col]),
    )

    # Update the size of the specified point if x_known and y_known are provided
    if x_known is not None and y_known is not None:
        point_index = find_point_index(x_col, y_col, x_known, y_known)
        if point_index is not None:
            session_state.plot_data.at[point_index, marker_col] = larger_point_size
            session_state.current_figure.update_traces(
                marker=dict(
                    size=session_state.plot_data[marker_col],
                    color=session_state.plot_data[color_col],
                )
            )


def create_umap_scatter(
    x_col,
    y_col,
    color_col,
    size=0.5,
    alpha=0.5,
    cmap="turbo",
    larger_point_size=4,
    marker_col="marker_size",
):
    """
    Update a UMAP scatter plot using Plotly Express.

    Parameters:
    - x_col: Column name or index for the x-axis.
    - y_col: Column name or index for the y-axis.
    - size: Default marker size for points.
    - alpha: Marker transparency.
    - color_col: Column name or index for the color of markers.
    - cmap: Colormap for coloring markers.
    - larger_point_size: Size of the larger point.
    - marker_col: Column name for the size column(to create).

    Returns:
    - Created Plotly Express figure.
    """
    session_state.plot_data[marker_col] = size

    session_state.current_figure = px.scatter(
        session_state.plot_data,
        x=x_col,
        y=y_col,
        size_max=larger_point_size,
        size=marker_col,
        opacity=alpha,
        color=color_col,
        color_continuous_scale=cmap,
        title=f"UMAP entropy ({session_state.plot_data.shape[0]} samples)",  # fix me {session_state.plot_data.shape[1]-3} dims,
    )
    # Update layout
    session_state.current_figure.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=True),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=True),
        autosize=True,
    )


def create_subplots(n_neighbors, select_label, target_cols, source_target_cols):
    subplots = []
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    dimred = session_state.plot_data[target_cols].values
    knn.fit(dimred)
    distance_mat, neighbors = knn.kneighbors(dimred[select_label, :].reshape(1, -1))

    entropies_sample_masked = session_state.plot_data.iloc[list(neighbors[0, :]), :]

    # extract columns except of source_target_cols
    dist = session_state.source_data.loc[
        :, ~session_state.source_data.columns.isin(source_target_cols)
    ]
    dist_reindex = dist.reset_index(drop=True)

    for i, motif_iloc in enumerate(entropies_sample_masked.index):
        motif = dist_reindex.iloc[
            motif_iloc : motif_iloc + session_state.num_features, :
        ].values
        motif = motif - motif[0, :]

        motif_x = np.arange(
            0, session_state.num_features / 10, 0.1
        )  ## resampled frame rate implied at 10 Hz

        lon_slice = session_state.source_data[source_target_cols[0]].iloc[
            motif_iloc : motif_iloc + session_state.num_features
        ]
        lat_slice = session_state.source_data[source_target_cols[1]].iloc[
            motif_iloc : motif_iloc + session_state.num_features
        ]
        lon_width = lon_slice.max() - lon_slice.min()
        lat_width = lat_slice.max() - lat_slice.min()
        square_width = 0.75 * max(lon_width, lat_width)
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=lon_slice,
                y=lat_slice,
                mode="markers",
                marker=dict(
                    size=5,
                    color=motif_x,
                    colorscale="turbo",
                ),
            )
        )
        fig.update_layout(
            xaxis=dict(
                range=[lon_slice.mean() - square_width, lon_slice.mean() + square_width]
            ),
            yaxis=dict(
                range=[lat_slice.mean() - square_width, lat_slice.mean() + square_width]
            ),
            showlegend=False,
            title=f"sample:{select_label}   motif:{motif_iloc}    {i+1}/{len(entropies_sample_masked)}",
        )
        subplots.append(fig)
    return subplots


interact_plot = st.empty()
if not session_state.source_path:
    h5_files = [
        os.path.join(SOURCE_DIR, file)
        for file in os.listdir(SOURCE_DIR)
        if file.endswith(".h5")
    ]
    with interact_plot.container():
        session_state.source_path = st.selectbox(
            "Choose file to process:",
            ("", *h5_files),
        )

        if (
            not isinstance(session_state.plot_data, pd.DataFrame)
            and session_state.source_path
        ):
            # if not os.path.exists(session_state.source_path):
            (
                session_state.plot_data,
                session_state.source_data,
                session_state.num_features,
            ) = create_output(session_state.source_path)

if len(session_state.latest_chosen_point) == 0 and isinstance(
    session_state.plot_data, pd.DataFrame
):
    create_umap_scatter(
        x_col=TARGET_COL_1,
        y_col=TARGET_COL_2,
        color_col=COLOR_COL,
    )
    with interact_plot.container():
        session_state.latest_chosen_point = plotly_events(
            session_state.current_figure, key="umap_scatter"
        )

if len(session_state.latest_chosen_point) > 0 and not session_state.plotting:
    chosen_x = session_state.latest_chosen_point[0]["x"]
    chosen_y = session_state.latest_chosen_point[0]["y"]
    update_umap_scatter(
        x_col=TARGET_COL_1,
        y_col=TARGET_COL_2,
        color_col=COLOR_COL,
        x_known=chosen_x,
        y_known=chosen_y,
    )
    with interact_plot.container():
        plotly_events(session_state.current_figure, key="chosen_point")
        st.write(f"You chose: ({chosen_x}, {chosen_y})")

if (
    st.button("Get plots") and len(session_state.latest_chosen_point) > 0
) or session_state.plotting:
    session_state.plotting = True
    with interact_plot.container():
        plotly_events(session_state.current_figure, key="get_subplots")
        session_state.n_neighbors = st.number_input(
            "Choose number of neighbors:", value=0
        )
        if session_state.n_neighbors > 0 and st.button("plot it!"):
            chosen_label = find_point_sequence(
                x_col=TARGET_COL_1,
                y_col=TARGET_COL_2,
                x_col_value=session_state.latest_chosen_point[0]["x"],
                y_col_value=session_state.latest_chosen_point[0]["y"],
            )
            plots = create_subplots(
                n_neighbors=session_state.n_neighbors,
                select_label=chosen_label,
                target_cols=[TARGET_COL_1, TARGET_COL_2],
                source_target_cols=[SOURCE_TARGET_COL_1, SOURCE_TARGET_COL_2],
            )
            for plot in plots:
                st.plotly_chart(plot)


if st.button("Change point") and session_state.current_figure is not None:
    session_state.latest_chosen_point = []
    session_state.plotting = False
    st.rerun()

if st.button("Change file"):
    session_state.latest_chosen_point = []
    session_state.plotting = False
    session_state.source_path = ""
    session_state.plot_data = None
    session_state.source_data = None
    st.rerun()
