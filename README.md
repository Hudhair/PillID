# PillID - Real-Time Medication Identification System

A computer vision application that uses a webcam to identify pills based on visual features including **shape, color, and imprint text**.

The project combines image processing, optical character recognition (OCR), and a SQLite database to match detected pill characteristics with stored medication records.

---

## Features

- Real-time webcam pill detection
- Pill shape recognition using contour analysis
- Color detection using HSV color classification
- Imprint text recognition using EasyOCR
- SQLite database for storing medication information
- Feature-based pill matching using:
  - Imprint text
  - Color
  - Shape

---

## Technologies Used

- Python
- OpenCV
- EasyOCR
- SQLite
- NumPy

---

## How It Works

1. The webcam captures a live image frame.
2. OpenCV processes the image and identifies the pill contour.
3. The application extracts visual features:
   - Shape
   - Color
   - Imprint text
4. EasyOCR reads the pill imprint.
5. Extracted features are compared against stored pill records in SQLite.
6. The program displays the detected medication information.
