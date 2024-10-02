# Face Recognition Attendance System

This project implements a face recognition attendance system using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm. The system detects faces in real-time from the webcam, recognizes individuals, and logs their attendance in an Excel file.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- Directory Structure
- Code Overview
  - 1. Face Image Collection
  - 2. Model Training
  - 3. Real-time Face Recognition and Attendance Logging
- Usage
- Attendance Logging
- Error Handling
- Notes
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
```
## License

MIT License

Copyright (c) 2024 Jitendra

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

