import numpy as np
import cv2
import sqlite3
from dataclasses import dataclass


try:
    import easyocr
    OCR_READER = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    OCR_READER = None
    print("[WARN] EasyOCR not available. Install with: pip install easyocr")
    print("       Error:", e)

# ---------------------------
# Performance knobs
# ---------------------------
OCR_EVERY_N_FRAMES = 15      # OCR runs once every 15
OCR_DOWNSCALE = 0.5          # Downscale imprint image before OCR
FRAME_COUNT = 0
LAST_IMPRINT_TEXT = ""

# ---------------------------
# DB setup
# ---------------------------
DB_PATH = "pills.db"


@dataclass
class PillMatch:
    drug: str
    strength: str
    score: int


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_name TEXT NOT NULL,
            strength TEXT,
            imprint TEXT,
            color TEXT,
            shape TEXT
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pills_imprint ON pills(imprint);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pills_color_shape ON pills(color, shape);")
    conn.commit()
    return conn


def add_pill(conn, drug_name, strength, imprint, color, shape):
    """Helper to seed the DB with known pills."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO pills (drug_name, strength, imprint, color, shape)
        VALUES (?, ?, ?, ?, ?)
    """, (drug_name, strength, imprint, color, shape))
    conn.commit()


def normalize_imprint(text: str) -> str:
    if not text:
        return ""
    t = text.upper().strip()
    t = "".join(ch for ch in t if ch.isalnum() or ch in "-/")
    return t


def find_best_match(conn, color: str, shape: str, imprint: str) -> PillMatch | None:
    """Imprint-first matching + fallback to (color+shape)."""
    imprint_n = normalize_imprint(imprint)
    cur = conn.cursor()
    candidates: list[PillMatch] = []

    # 1) Strong match: imprint exact
    if imprint_n:
        cur.execute("""
            SELECT drug_name, strength, imprint, color, shape
            FROM pills
            WHERE UPPER(imprint) = ?
        """, (imprint_n,))
        for drug, strength, imp, col, shp in cur.fetchall():
            score = 100
            if col and color and col == color:
                score += 10
            if shp and shape and shp == shape:
                score += 10
            candidates.append(PillMatch(drug=drug, strength=strength or "", score=score))

    # 2) Fallback: color + shape
    if not candidates and color and shape:
        cur.execute("""
            SELECT drug_name, strength
            FROM pills
            WHERE color = ? AND shape = ?
        """, (color, shape))
        for drug, strength in cur.fetchall():
            candidates.append(PillMatch(drug=drug, strength=strength or "", score=50))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[0]


def draw_hud(frame, color, shape, imprint, match: PillMatch | None):
    """Draws static text in the corner of the frame."""
    x, y = 20, 30
    dy = 28

    drug_text = match.drug if match else "Unknown"
    strength_text = match.strength if (match and match.strength) else ""

    lines = [
        f'Color: {color or "Unknown"}',
        f'Shape: {shape or "Unknown"}',
        f'Text:  {imprint or "Unknown"}',
        f'Drug:  {(" ".join([drug_text, strength_text])).strip()}'
    ]

    # Simple "shadow" for readability
    for i, line in enumerate(lines):
        yy = y + i * dy
        cv2.putText(frame, line, (x + 1, yy + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, line, (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def detect_color_hsv(frame_bgr, contour):
    """Return a simple color name for the region inside `contour`."""
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255]
    if pixels.size == 0:
        return "Unknown"

    h_med, s_med, v_med = np.median(pixels, axis=0)

    if v_med < 60:
        return "Black/Dark"
    if s_med < 40 and v_med > 170:
        return "White"
    if s_med < 40:
        return "Gray"

    h = h_med
    if h < 10 or h >= 170:
        return "Red"
    if 10 <= h < 20:
        return "Orange"
    if 20 <= h < 35:
        return "Yellow"
    if 35 <= h < 85:
        return "Green"
    if 85 <= h < 125:
        return "Blue"
    if 125 <= h < 170:
        return "Purple"
    return "Unknown"


# ---------------------------
# Imprint extraction + preprocessing
# ---------------------------
def extract_pill_roi(frame, contour, pad=10):
    """Crop ROI around pill contour + return masked ROI and mask."""
    x, y, w, h = cv2.boundingRect(contour)
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, frame.shape[1])
    y1 = min(y + h + pad, frame.shape[0])

    roi = frame[y0:y1, x0:x1].copy()

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    shifted = contour - [x0, y0]
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)

    roi_masked = cv2.bitwise_and(roi, roi, mask=mask)
    return roi_masked, mask, (x0, y0, x1, y1)


def preprocess_imprint(roi_bgr, mask):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.addWeighted(top, 1.0, black, 1.0, 0)

    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    bin_img = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k2, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k2, iterations=1)

    return bin_img


def ocr_imprint(bin_img, scale=0.5):
    """OCR the processed imprint image. Uses caching to reduce jitter."""
    global LAST_IMPRINT_TEXT

    if OCR_READER is None:
        return ""

    # Downscale for speed
    if scale is not None and 0 < scale < 1.0:
        h, w = bin_img.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        bin_img = cv2.resize(bin_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    results = OCR_READER.readtext(bin_img, detail=0)
    text = " ".join(results).strip()
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in "-/")

    if cleaned:
        LAST_IMPRINT_TEXT = cleaned

    return LAST_IMPRINT_TEXT


# ---------------------------
# Shape detection + DB match + HUD
# ---------------------------
def shapeDetection(frame, conn, debug=False):
    global FRAME_COUNT, LAST_IMPRINT_TEXT
    FRAME_COUNT += 1
    do_ocr = (FRAME_COUNT % OCR_EVERY_N_FRAMES == 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Uncomment if your background becomes the contour instead of the pill:
    # thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    # If no contours, still show HUD as unknown
    if not contours:
        draw_hud(frame, "", "", LAST_IMPRINT_TEXT, None)
        return frame, thresh

    # Use the largest contour as "the pill" for stable HUD
    pill_contour = max(contours, key=cv2.contourArea)

    detected_shape = "Unknown"
    detected_color = "Unknown"
    detected_imprint = LAST_IMPRINT_TEXT

    # Optionally draw all contours, but compute features primarily on the pill contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 800:
            continue

        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

    # Compute features on the pill contour
    area = cv2.contourArea(pill_contour)
    if area >= 800:
        peri = cv2.arcLength(pill_contour, True)
        approx = cv2.approxPolyDP(pill_contour, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)

        sides = len(approx)
        if sides == 3:
            detected_shape = "Triangle"
        elif sides == 4:
            ar = w / float(h)
            detected_shape = "Square" if 0.90 <= ar <= 1.10 else "Rectangle"
        else:
            circularity = 4 * np.pi * (area / (peri * peri + 1e-9))
            detected_shape = "Circle" if circularity > 0.80 else "Oval/Other"

        detected_color = detect_color_hsv(frame, pill_contour)

        if do_ocr:
            roi, mask, _ = extract_pill_roi(frame, pill_contour, pad=10)
            bin_imprint = preprocess_imprint(roi, mask)
            detected_imprint = ocr_imprint(bin_imprint, scale=OCR_DOWNSCALE)

            if debug:
                cv2.imshow("pill_roi", roi)
                cv2.imshow("imprint_bin", bin_imprint)

    # DB match
    match = find_best_match(conn, detected_color, detected_shape, detected_imprint)

    # HUD overlay (static text)
    draw_hud(frame, detected_color, detected_shape, detected_imprint, match)

    return frame, thresh


# ---------------------------
# Main
# ---------------------------
conn = init_db()

# OPTIONAL: seed examples (run once, then comment these out)
# add_pill(conn, "Acetaminophen", "500 mg", "L484", "White", "Oval/Other")
# add_pill(conn, "Acetaminophen", "325 mg", "TYLENOL", "White", "Oval/Other")
# add_pill(conn, "Ibuprofen", "200 mg", "I-2", "Orange", "Oval/Other")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated, thresh = shapeDetection(frame, conn=conn, debug=False)

    cv2.imshow("frame", annotated)
    cv2.imshow("thresh", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
