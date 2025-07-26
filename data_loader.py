# data_loader.py
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_dlc_data(csv_path):
    """
    Loads and processes a DeepLabCut CSV file using a robust, vectorized method.
    """
    logger.info(f"Loading DeepLabCut data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=[1, 2, 3], index_col=0)
    except FileNotFoundError:
        logger.error(f"Data file not found at: {csv_path}")
        raise

    df.index.name = 'frame'
    df.replace(-1, np.nan, inplace=True)

    # Use future_stack=True for modern, consistent behavior and to silence warnings.
    long_df = df.stack(level=['individuals', 'bodyparts', 'coords'], future_stack=True)
    unstacked_df = long_df.unstack(level='coords')
    final_df = unstacked_df.reset_index()

    pivoted_df = final_df.pivot(
        index=['frame', 'individuals'],
        columns='bodyparts',
        values=['x', 'y', 'likelihood']
    )

    pivoted_df.columns = [f'{coord}_{bp}' for coord, bp in pivoted_df.columns]
    pivoted_df.reset_index(inplace=True)

    pivoted_df.rename(columns={'individuals': 'track_id'}, inplace=True)
    pivoted_df['track_id'] = pivoted_df['track_id'].str.replace('animal', '').astype(int)

    rename_map = {}
    all_bodyparts_snake = {col.split('_', 1)[1] for col in pivoted_df.columns if '_' in col}

    for bp_snake in all_bodyparts_snake:
        bp_pascal = ''.join(word.capitalize() for word in bp_snake.split('_'))
        rename_map[f'x_{bp_snake}'] = f'{bp_pascal}_x'
        rename_map[f'y_{bp_snake}'] = f'{bp_pascal}_y'
        rename_map[f'likelihood_{bp_snake}'] = f'{bp_pascal}_conf'
    pivoted_df.rename(columns=rename_map, inplace=True)
    
    center_cols_x = [col for col in pivoted_df.columns if col.endswith('_x')]
    pivoted_df['center_x'] = pivoted_df[center_cols_x].mean(axis=1)
    center_cols_y = [col for col in pivoted_df.columns if col.endswith('_y')]
    pivoted_df['center_y'] = pivoted_df[center_cols_y].mean(axis=1)

    if 'MouseCenter_x' in pivoted_df.columns:
        pivoted_df['center_x'] = pivoted_df['MouseCenter_x'].fillna(pivoted_df['center_x'])
        pivoted_df['center_y'] = pivoted_df['MouseCenter_y'].fillna(pivoted_df['center_y'])

    logger.info(f"Successfully loaded and processed {len(pivoted_df)} rows of data.")
    return pivoted_df