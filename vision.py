import cv2
import numpy as np

sigma = 1.0   # smoothing sigma
mode = 'o'    # default mode = original

def apply_gaussian_blur(frame, sigma):
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(frame, (k, k), sigma)

def sobel_x(img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

def sobel_y(img):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

def sobel_magnitude(gx, gy):
    mag = np.sqrt(gx**2 + gy**2)
    mag = np.uint8(255 * mag / np.max(mag))
    return mag

def sobel_threshold(img):
    gx = sobel_x(img)
    gy = sobel_y(img)
    mag = sobel_magnitude(gx, gy)
    _, th = cv2.threshold(mag, 80, 255, cv2.THRESH_BINARY)
    return th

def laplacian_of_gaussian(img, sigma):
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(img, (k, k), sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log = cv2.convertScaleAbs(log)
    return log

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

print("=======================================")
print(" Press keys to switch between modes:")
print(" o → Original frame")
print(" x → Sobel X")
print(" y → Sobel Y")
print(" m → Sobel Magnitude")
print(" s → Sobel + Threshold")
print(" l → LoG (Laplacian of Gaussian)")
print(" + → Increase sigma")
print(" - → Decrease sigma")
print(" q → Quit")
print("=======================================")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === APPLY GAUSSIAN BLUR ON EVERY FRAME ===
    blurred = apply_gaussian_blur(gray, sigma)

    # === SELECT OPERATION BASED ON MODE ===
    if mode == 'o':
        output = frame

    elif mode == 'x':
        gx = sobel_x(blurred)
        output = cv2.convertScaleAbs(gx)

    elif mode == 'y':
        gy = sobel_y(blurred)
        output = cv2.convertScaleAbs(gy)

    elif mode == 'm':
        gx = sobel_x(blurred)
        gy = sobel_y(blurred)
        output = sobel_magnitude(gx, gy)

    elif mode == 's':
        output = sobel_threshold(blurred)

    elif mode == 'l':
        output = laplacian_of_gaussian(gray, sigma)

    else:
        output = frame

    cv2.putText(output, f"Mode: {mode} | Sigma={sigma:.1f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("CV Project Output", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('o'):
        mode = 'o'

    elif key == ord('x'):
        mode = 'x'

    elif key == ord('y'):
        mode = 'y'

    elif key == ord('m'):
        mode = 'm'

    elif key == ord('s'):
        mode = 's'

    elif key == ord('l'):
        mode = 'l'

    elif key == ord('+'):
        sigma += 0.5
        if sigma > 10: sigma = 10

    elif key == ord('-'):
        sigma -= 0.5
        if sigma < 0.5: sigma = 0.5

cap.release()
cv2.destroyAllWindows()
