import cv2
import os
import sys

def register_student(roll, name):
    cap = cv2.VideoCapture(0)
    print("📷 Press 's' to capture photo")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture Face - Press 's'", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = f"{roll}_{name}.jpg"
            path = os.path.join('student_images', filename)
            cv2.imwrite(path, frame)
            print(f"✅ Photo saved as {path}")
            break
        elif key == ord('q'):
            print("❌ Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    roll = sys.argv[1]
    name = sys.argv[2]
    register_student(roll, name)
