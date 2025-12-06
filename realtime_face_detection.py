import cv2
import time

def main():
    # Load the Haar cascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Could not load Haar cascade XML file.")
        return

    # Open the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert to grayscale because Haar cascade works on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)  # You can tune this
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Put text on the frame: FPS and face count
        text_fps = f"FPS: {fps:.2f}"
        text_faces = f"Faces: {len(faces)}"
        cv2.putText(frame, text_fps, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, text_faces, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the output
        cv2.imshow("Real-time Face Detection - Press q to quit", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
