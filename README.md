[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/farhanaugustine/Gait_Analysis_DeepLabCut)

# Gait_Analysis_DeepLabCut
Configurable Python project for analyzing gait in animal video experiments.

This is a Python-based tool for detailed analysis of animal behavior in videos, focusing tests like open-field. It leverages modern pose-estimation data from models like DeepLabCut to generate a video dashboard that visualizes key metrics including posture dynamics, spatial exploration, and stride/gait analysis.

Originally inspired by stride-level analysis methods (See Acknowledgements), this toolkit is a flexible and powerful solution for researchers using DeepLabCut's SuperAnimal models, providing quantitative insights into stride details.

<img width="1327" height="551" alt="Screenshot 2025-07-26 182412" src="https://github.com/user-attachments/assets/fbe36f9d-e8e4-4f0c-98ce-dff84319379f" />



https://github.com/user-attachments/assets/5793b287-abc7-4465-903e-67249fe1d39d


## Features

- **Dynamic Video Dashboard**: Combines the source video with a rich, multi-panel dashboard showing live metrics.
- **Advanced Gait Analysis**: Calculates stride and step-level metrics (length, speed, width) and visualizes paw phases in a Hildebrand-style diagram.
- **Pose & Posture Metrics**: Quantifies body elongation, turning speed, and overall body angle on a frame-by-frame basis.
- **Region of Interest (ROI) Analysis**: Allows users to interactively draw named ROIs and automatically calculates time spent and entries for each zone.
- **Flexible Configuration**: A centralized `config.py` file makes it easy to adapt the analysis to different videos, keypoint models, and parameters.
- **Exportable Data**: Saves all calculated metrics to a detailed CSV file for further statistical analysis.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/farhanaugustine/Gait_Analysis_DeepLabCut.git
   cd Gait_Analysis_DeepLabCut
   ```

2. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install pandas numpy opencv-python tqdm
   ```

## How to Use

### Step 1: Place Your Files
Place your input video file (e.g., `.mp4`) and the corresponding pose-estimation CSV file from DeepLabCut into the main project directory.

### Step 2: Configure the Analysis
Open the `config.py` file and update the parameters to match your setup. Key settings include file paths and keypoint definitions. See the [Configuration](#configuration-configpy) section for details.

### Step 3: Run the Main Script
Execute the main analysis script from your terminal.
```bash
python main.py
```

### Step 4: Draw Regions of Interest (ROIs)
If no `roi_config.json` file is found, the script will prompt you to draw ROIs on the first frame of the video:
- An OpenCV window will appear showing the first frame.
- Enter a name for your first ROI in the console (e.g., "Center").
- Left-click on the image to place points for the ROI polygon.
- Press the `c` key to confirm and save the current ROI.
- Press the `r` key to reset the points for the current ROI.
- After saving an ROI, enter a name for the next one. Press `Enter` without typing a name to finish.
- ROI definitions will be saved to `roi_config.json` for future runs.

The script will process the video and generate output files in the `results/` directory.

## Analysis Metrics Explained

Behavior Analysis Dashboard provides a multi-faceted analysis of behavior, broken down into several key categories.

### Pose Metrics
These metrics describe the animal's posture and orientation.

| Metric              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Body Speed          | Speed of the animal's center point, measured in pixels per frame.           |
| Elongation          | Distance between Nose and TailBase keypoints, proxy for body stretching.    |
| Posture Variability | Standard deviation of Elongation over a rolling window.                    |
| Body Angle          | Orientation of the body axis (Neck to Nose), measured in degrees.           |
| Turning Speed       | Rate of change of Body Angle, measured in degrees per frame.                |

### Gait Analysis: Stride & Step
These metrics provide a detailed look at locomotor patterns.

| Metric         | Definition                                                                 |
|----------------|----------------------------------------------------------------------------|
| Stride         | Gait cycle from one foot-strike of the reference paw to the next.          |
| Step Length    | Distance from reference paw's foot-strike to the opposing paw's foot-strike.|
| Step Width     | Perpendicular distance from opposing paw's foot-strike to the line of progression. |
| Stride Speed   | Average body speed during a single stride cycle.                           |

#### Paw Phase
The Hildebrand diagram illustrates two paw states based on speed relative to a threshold.

| Phase        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| Stance Phase | Paw on the ground, speed below `PAW_SPEED_THRESHOLD_PX_PER_FRAME`. (Green) |
| Swing Phase  | Paw lifted, speed above threshold. (Red)                                   |

### ROI Analysis
Quantifies spatial exploration based on user-defined zones.

| Metric        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| Time in ROI   | Total time (seconds) the animal's center point spends in each ROI.          |
| ROI Entries   | Number of times the animal enters each ROI.                                 |

## Configuration (`config.py`)

All user-configurable parameters are in `config.py`.

### Section 1: File Paths

| Parameter            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `INPUT_VIDEO_PATH`   | Path to input video file (e.g., `.mp4`).                                    |
| `INPUT_CSV_PATH`     | Path to pose-estimation CSV file from DeepLabCut.                           |
| `ROI_CONFIG_PATH`    | Filename for ROI definitions. Created if it doesn't exist.                  |
| `OUTPUT_CSV_PATH`    | Path for CSV file with all calculated data.                                 |
| `GAIT_ANALYSIS_PATH` | Path for CSV file summarizing detected strides.                             |
| `OUTPUT_VIDEO_PATH`  | Path for rendered analysis video.                                           |

### Section 2 & 4: Dataset and Skeleton Configuration

| Parameter                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `KEYPOINT_ORDER`         | List of keypoint names as they appear in the model's output.                |
| `SKELETON_CONNECTIONS`   | List of tuples defining connections between keypoints for skeleton drawing. |
| `ELONGATION_CONNECTION`  | Tuple of two keypoint names for body elongation calculation.                |
| `BODY_ANGLE_CONNECTION`  | Tuple of two keypoint names for body angle calculation.                     |

### Section 3: Gait Analysis Parameters

| Parameter                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| `GAIT_PAWS`                        | List of keypoint names for gait analysis (e.g., paws or proxies like LeftHip). |
| `PAW_ORDER_HILDEBRAND`             | Order of paws/proxies on the Hildebrand gait diagram.                       |
| `PAW_SPEED_THRESHOLD_PX_PER_FRAME` | Speed (pixels/frame) below which a paw is in "stance" phase.                |
| `STRIDE_REFERENCE_PAW`             | Keypoint name defining start/end of a stride cycle.                         |

### Section 5 & 6: Display and Analysis Parameters

| Parameter                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `RESIZED_VIDEO_WIDTH`      | Width (pixels) of the video panel in the output.                            |
| `DASHBOARD_WIDTH`          | Width (pixels) of the dashboard panel.                                      |
| `DETECTION_CONF_THRESHOLD` | Minimum likelihood score from DeepLabCut for a valid keypoint.              |

## Acknowledgements

This project builds upon methodologies for behavioral analysis and was inspired by:

- **Title**: Stride-level analysis of mouse open field behavior using deep-learning-based pose estimation  
- **Authors**: Keith Sheppard, et al.  
- **Journal**: Cell Reports (2021)  
- **DOI**: [10.1016/j.celrep.2021.110231](https://doi.org/10.1016/j.celrep.2021.110231)

Behavior Analysis Dashboard relies on open-source projects, including DeepLabCut, OpenCV, and the scientific Python ecosystem (`pandas`, `NumPy`).
