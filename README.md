# Face Recognition Attendance System

This project implements a face recognition attendance system using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm. The system detects faces in real-time from the webcam, recognizes individuals, and logs their attendance in an Excel file.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Code Overview](#code-overview)
  - [1. Face Image Collection](#1-face-image-collection)
  - [2. Model Training](#2-model-training)
  - [3. Real-time Face Recognition and Attendance Logging](#3-real-time-face-recognition-and-attendance-logging)
- [Usage](#usage)
- [Attendance Logging](#attendance-logging)
- [Error Handling](#error-handling)
- [Notes](#notes)
- [License](#license)

## Features

- Real-time face detection and recognition using a webcam.
- Attendance logging in an Excel file.
- Confidence score for face recognition.
- User-friendly interface with feedback on recognized faces.

## Requirements

Ensure you have the following packages installed:

- Python 3.x
- OpenCV (`opencv-python` and `opencv-contrib-python` for the face module)
- NumPy
- Pandas
- `openpyxl` (for reading/writing Excel files)

You can install the necessary packages using the following command:

```bash
pip install opencv-python opencv-contrib-python numpy pandas openpyxl
