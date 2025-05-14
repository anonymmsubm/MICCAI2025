# MICCAI2025

Anonymous repository for the MICCAI 2025 submission.

This study aims to develop and validate a transformer-based algorithm that predicts radiological errors in chest X-ray interpretation by analyzing longitudinal gaze patterns and image data. 

## Repository Structure

```
├── data/                  # Data loading and preprocessing scripts
├── requirements.txt       # Python dependencies
└── README.md              # Project overview
```


## Getting started

You will need two types of data: fixation points (x, y) and chest X-ray images. If you have raw gaze data, it needs to be processed to identify fixations (i.e., moments when the gaze is relatively stable), as the number of (x, y) points in a file will affect computation time. You may use any fixation detection method. One commonly cited approach is:

> Dario D. Salvucci and Joseph H. Goldberg (2000). Identifying Fixations and Saccades in Eye-Tracking Protocols. 


