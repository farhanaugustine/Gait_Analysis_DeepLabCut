# main.py
import cv2
import numpy as np
import pandas as pd
import os
import logging
from collections import defaultdict
from tqdm import tqdm

import config
from data_loader import load_dlc_data
from analysis import (
    process_data,
    calculate_roi_event_timeline
)
from dashboard import Dashboard
from utils import get_rois, build_skeleton_indices, draw_skeleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def render_video(df, gait_df, config, rois):
    logger.info("Starting video rendering process...")
    try:
        cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        max_frame = int(df['frame'].max()) if not df.empty else 0
        cap.release()
    except Exception as e:
        logger.error(f"Failed to initialize video properties: {e}", exc_info=True)
        raise

    roi_event_timeline = calculate_roi_event_timeline(df)
    frame_data_map = {frame: group for frame, group in df.groupby('frame')}
    stride_end_map = {}
    if gait_df is not None and not gait_df.empty:
        for _, stride in gait_df.iterrows():
            stride_end_map[stride['end_frame']] = stride.to_dict()

    video_w, dash_w = config.RESIZED_VIDEO_WIDTH, config.DASHBOARD_WIDTH
    out_w, out_h = video_w + dash_w, h
    scale_x, scale_y = video_w / w, out_h / h

    cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
    out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    dashboard = Dashboard(config, video_height=out_h, fps=fps)
    skeleton_indices = build_skeleton_indices(config.KEYPOINT_ORDER, config.SKELETON_CONNECTIONS)
    roi_stats = defaultdict(lambda: {'time_s': 0, 'entries': 0})

    logger.info("Starting frame-by-frame rendering with new dashboard...")
    for frame_number in tqdm(range(max_frame + 1), desc="Rendering Video"):
        success, frame = cap.read()
        if not success: frame = np.zeros((h, w, 3), dtype=np.uint8)
        if frame_number in roi_event_timeline:
            for event in roi_event_timeline[frame_number]:
                if event['type'] == 'entry': roi_stats[event['roi_name']]['entries'] += 1

        animals_on_frame_df = frame_data_map.get(frame_number, pd.DataFrame())
        animals_on_frame = [] if animals_on_frame_df.empty else animals_on_frame_df.to_dict('records')

        for animal in animals_on_frame:
            roi_name = animal.get('current_roi')
            if roi_name and roi_name != 'None': roi_stats[roi_name]['time_s'] += 1 / fps

        speed_values = [a['speed'] for a in animals_on_frame if pd.notna(a.get('speed'))]
        posture_values = [a['posture_variability'] for a in animals_on_frame if pd.notna(a.get('posture_variability'))]
        
        stats_for_drawing = {
            'animals_on_frame': animals_on_frame,
            'speed_mean': np.mean(speed_values) if speed_values else 0,
            'posture_mean': np.mean(posture_values) if posture_values else 0,
            'newly_completed_stride': stride_end_map.get(frame_number),
            'roi_stats': roi_stats,
        }

        resized_frame = cv2.resize(frame, (video_w, out_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[0:out_h, 0:video_w] = resized_frame

        for roi in rois:
            scaled_coords = (roi['coords'] * [scale_x, scale_y]).astype(np.int32)
            cv2.polylines(canvas, [scaled_coords], True, (255, 0, 0), 2)

        for animal in animals_on_frame:
            keypoints = np.array([[animal.get(f'{name}_x'), animal.get(f'{name}_y')] for name in config.KEYPOINT_ORDER], dtype=np.float32)
            scaled_keypoints = keypoints * [scale_x, scale_y]
            draw_skeleton(canvas, scaled_keypoints, skeleton_indices, config.KEYPOINT_COLOR, config.SKELETON_COLOR, config.KEYPOINT_RADIUS)

        canvas = dashboard.update_and_draw(canvas, stats_for_drawing, frame_number)
        out.write(canvas)
        
    cap.release()
    out.release()
    logger.info(f"Video rendering complete. Saved to {config.OUTPUT_VIDEO_PATH}")

def run():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    rois = get_rois(config.INPUT_VIDEO_PATH, config.ROI_CONFIG_PATH)
    raw_df = load_dlc_data(config.INPUT_CSV_PATH)
    if raw_df.empty:
        logger.error("Failed to load any data from the CSV file. Exiting.")
        return

    logger.info("Consolidating fragmented tracks for single-animal analysis...")
    raw_df = raw_df[raw_df['track_id'] == 0].copy()
    raw_df['track_id'] = 1
    logger.info("Track consolidation complete. Using data for the first detected animal (track_id=0).")

    final_df, gait_df = process_data(raw_df, rois)
    final_df.to_csv(config.OUTPUT_CSV_PATH, index=False)
    logger.info(f"Saved final processed data to {config.OUTPUT_CSV_PATH}")

    if gait_df is not None and not gait_df.empty:
        gait_df.to_csv(config.GAIT_ANALYSIS_PATH, index=False)
        logger.info(f"Saved gait analysis summary to {config.GAIT_ANALYSIS_PATH}")

        logger.info("Using reliable gait data for stride video generation.")
        strides_for_videos = gait_df.rename(columns={'start_frame': 'stride_start_frame', 'end_frame': 'stride_end_frame'})
        stride_output_path = os.path.join(config.RESULTS_DIR, 'custom_filtered_strides.csv')
        columns_to_save = ['track_id', 'stride_start_frame', 'stride_end_frame']
        if all(c in strides_for_videos.columns for c in columns_to_save):
            strides_for_videos[columns_to_save].to_csv(stride_output_path, index=False)
            logger.info(f"Saved reliable stride data for video generation to {stride_output_path}")
    else:
        logger.warning("No gait data was generated. Stride video generation will be skipped.")

    render_video(final_df, gait_df, config, rois)
    logger.info("Analysis complete.")

if __name__ == "__main__":
    run()