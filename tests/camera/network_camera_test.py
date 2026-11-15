
import cv2

stream_url = "http://192.168.1.172:5000/video"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Cannot connect to laptop camera")
    exit()

print("Connected to laptop camera!")

while True:
    success, frame = cap.read()

    if not success:
        print("Cannot receive video")
        break

    cv2.imshow("Laptop Webcam - Desktop", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
