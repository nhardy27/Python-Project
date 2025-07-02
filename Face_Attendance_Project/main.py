import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
import pytz
import pandas as pd

path = 'student_images'
images = []
student_names = []
student_ids = []

for filename in os.listdir(path):
    if filename.endswith(('.jpg', '.png')):
        img = cv2.imread(f'{path}/{filename}')
        images.append(img)
        roll_no, name = filename.split('.')[0].split('_')
        student_ids.append(roll_no)
        student_names.append(name)

def encode_faces(images):
    encoded_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encoded_list.append(encode)
        except:
            pass
    return encoded_list

known_encodings = encode_faces(images)

def mark_attendance(roll_no, name):
    india_time = datetime.now(pytz.timezone('Asia/Kolkata'))
    date = india_time.strftime('%Y-%m-%d')
    time = india_time.strftime('%H:%M:%S')

    folder = 'Attendance'
    os.makedirs(folder, exist_ok=True)

    file = f'{folder}/Attendance_{date}.csv'

    # Create file with header if it doesn't exist
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write('Roll No,Name,Date,Time\n')

    try:
        df = pd.read_csv(file, dtype={'Roll No': str})
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=['Roll No', 'Name', 'Date', 'Time'])

    roll_no_str = str(roll_no)

    # Check if already marked today
    already_marked = df[(df['Roll No'] == roll_no_str) & (df['Date'] == date)]

    if not already_marked.empty:
        print(f"⏳ Already marked today: {roll_no_str} - {name}")
    else:
        new_row = {'Roll No': roll_no_str, 'Name': name, 'Date': date, 'Time': time}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(file, index=False)
        print(f"✅ Attendance marked for {roll_no_str} - {name}")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

    faces_cur_frame = face_recognition.face_locations(rgb_img)
    encodes_cur_frame = face_recognition.face_encodings(rgb_img, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(known_encodings, encode_face)
        face_dist = face_recognition.face_distance(known_encodings, encode_face)
        match_index = np.argmin(face_dist)

        if matches[match_index]:
            name = student_names[match_index]
            roll = student_ids[match_index]
            mark_attendance(roll, name)

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f'{roll} {name}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Attendance System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
