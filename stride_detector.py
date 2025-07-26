# stride_detector.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

def _find_movement_tracks(track_df, speed_threshold=5):
    """
    Determines intervals where the mouse is moving at sufficient speed.
    These are called 'tracks'.
    """
    # Threshold for base of tail speed to find frames with movement
    track_df['in_track'] = track_df['tail_base_speed'] >= speed_threshold
    
    # Find start and end points of continuous movement blocks
    track_df['track_change'] = track_df['in_track'].diff().fillna(False)
    change_indices = track_df.index[track_df['track_change']]
    
    track_intervals = []
    is_moving = False
    start_frame = None

    for idx, row in track_df.iterrows():
        if row['in_track'] and not is_moving:
            start_frame = row['frame']
            is_moving = True
        elif not row['in_track'] and is_moving:
            track_intervals.append((start_frame, row['frame'] - 1))
            is_moving = False
    
    # Add the last track if the video ends during movement
    if is_moving:
        track_intervals.append((start_frame, track_df['frame'].iloc[-1]))
        
    return track_intervals

def _detect_steps_for_paw(paw_df, paw_name, body_speed_series, peak_speed_threshold=15):
    """
    Identifies individual steps for a given paw using peak detection on paw speed.
    """
    paw_speed = paw_df[f'{paw_name}_speed'].values
    # Use peak detection to find local maxima in speed
    peaks, _ = find_peaks(paw_speed)
    # Find minima by inverting the signal
    troughs, _ = find_peaks(-paw_speed)
    
    valid_steps = []
    for peak_idx in peaks:
        # Filter step based on peak speed
        animal_speed = body_speed_series.iloc[peak_idx]
        speed_filter = max(peak_speed_threshold, animal_speed if pd.notna(animal_speed) else 0)
        
        if paw_speed[peak_idx] < speed_filter:
            continue

        # Find the surrounding local minima (toe-off and foot-strike)
        pre_troughs = troughs[troughs < peak_idx]
        post_troughs = troughs[troughs > peak_idx]

        if pre_troughs.size > 0 and post_troughs.size > 0:
            toe_off_idx = pre_troughs[-1]
            foot_strike_idx = post_troughs[0]
            
            valid_steps.append({
                'start_frame': paw_df['frame'].iloc[toe_off_idx],
                'end_frame': paw_df['frame'].iloc[foot_strike_idx],
                'peak_frame': paw_df['frame'].iloc[peak_idx],
                'peak_speed': paw_speed[peak_idx]
            })
            
    return sorted(valid_steps, key=lambda x: x['start_frame'])

def detect_and_filter_strides(df):
    """
    Main function to run the complete stride detection and filtering process.
    Takes a fully processed DataFrame as input.
    """
    logger.info("Starting new stride detection and filtering process...")
    
    # Ensure necessary speed columns exist
    if 'tail_base_speed' not in df.columns:
        df['tail_base_speed'] = np.sqrt(df.groupby('track_id')['Base of Tail_x'].diff()**2 + df.groupby('track_id')['Base of Tail_y'].diff()**2)
    for paw in ['Left Rear Paw', 'Right Rear Paw']:
        if f'{paw}_speed' not in df.columns:
            df[f'{paw}_speed'] = np.sqrt(df.groupby('track_id')[f'{paw}_x'].diff()**2 + df.groupby('track_id')[f'{paw}_y'].diff()**2)
    
    all_valid_strides = []
    
    for track_id, animal_df in df.groupby('track_id'):
        logger.info(f"Processing animal track ID: {track_id}")
        
        # 1. Determine track intervals based on movement speed
        movement_tracks = _find_movement_tracks(animal_df.copy())
        
        unfiltered_strides_per_track = []

        for track_start, track_end in movement_tracks:
            track_df = animal_df[(animal_df['frame'] >= track_start) & (animal_df['frame'] <= track_end)]
            if track_df.empty:
                continue

            # 2. Identify individual steps for left and right hind paws
            left_steps = _detect_steps_for_paw(track_df, 'Left Rear Paw', track_df['tail_base_speed'])
            right_steps = _detect_steps_for_paw(track_df, 'Right Rear Paw', track_df['tail_base_speed'])

            if not left_steps:
                continue
            
            # 3. Group steps into strides using left paw as the delimiter
            potential_strides = []
            
            # Define stride intervals based on left paw steps
            stride_start = left_steps[0]['start_frame']
            for i, l_step in enumerate(left_steps):
                stride_end = l_step['end_frame']
                
                # Associate right hind paw step if it completes within the stride interval
                found_r_step = None
                for r_step in right_steps:
                    if stride_start <= r_step['end_frame'] <= stride_end:
                        found_r_step = r_step
                        break # Found a valid right step, associate it
                
                if found_r_step:
                    potential_strides.append({
                        'track_id': track_id,
                        'stride_start_frame': stride_start,
                        'stride_end_frame': stride_end,
                        'left_step_data': l_step,
                        'right_step_data': found_r_step,
                    })

                # The next stride begins right after the current one ends
                stride_start = stride_end + 1
            
            unfiltered_strides_per_track.append(potential_strides)

        # 4. Apply final aggressive filtering
        for stride_list in unfiltered_strides_per_track:
            if len(stride_list) <= 2:
                continue # Not enough strides to remove start/end and have any left
            
            # Remove the first and last strides of the track
            strides_to_check = stride_list[1:-1]
            
            # Keypoints for confidence check
            conf_keypoints = ['Nose', 'Base of Neck', 'Center Spine', 'Base of Tail', 
                              'Left Rear Paw', 'Right Rear Paw', 'Mid Tail', 'Tail Tip']
            conf_cols = [f'{kp}_conf' for kp in conf_keypoints]

            for stride in strides_to_check:
                stride_frames_df = df[
                    (df['frame'] >= stride['stride_start_frame']) &
                    (df['frame'] <= stride['stride_end_frame']) &
                    (df['track_id'] == stride['track_id'])
                ]
                
                # Discard stride if any keypoint confidence is below 0.3
                min_confidence = stride_frames_df[conf_cols].min().min()
                if min_confidence >= 0.3:
                    all_valid_strides.append(stride)

    logger.info(f"Found {len(all_valid_strides)} high-quality strides after filtering.")
    return pd.DataFrame(all_valid_strides)


if __name__ == '__main__':
    # --- Example Usage ---
    # This shows how you could integrate this script into your main workflow
    
    logger.info("Running stride_detector.py as a standalone script for demonstration.")
    
    # 1. Load your data (this assumes you've run the main analysis first)
    try:
        # You would typically load the full, processed DataFrame here
        # For this example, we'll try to load the CSV generated by main.py
        processed_df = pd.read_csv(os.path.join('results', 'final_analysis_data.csv'))
    except FileNotFoundError:
        logger.error("Could not find 'final_analysis_data.csv'.")
        logger.error("Please run main.py first to generate the necessary data file.")
        exit()

    # 2. Run the stride detection and filtering
    filtered_strides_df = detect_and_filter_strides(processed_df)
    
    # 3. Save the results
    if not filtered_strides_df.empty:
        output_path = os.path.join('results', 'custom_filtered_strides.csv')
        filtered_strides_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(filtered_strides_df)} filtered strides to {output_path}")