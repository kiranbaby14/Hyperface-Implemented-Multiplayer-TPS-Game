import cv2
import numpy as np
import dlib
import pyautogui
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from scipy.spatial import distance

model = load_model('gender_detection.model')
gender = True
count = 0

cap = cv2.VideoCapture(0)

classes = ['man', 'woman']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Distance Calculation Part---------------------------------------
def calculate_Distance(dis):
    a = distance.euclidean(dis[1], dis[5])
    b = distance.euclidean(dis[2], dis[4])
    c = distance.euclidean(dis[0], dis[3])
    mouth_aspect_ratio = (a + b) / (2.0 * c)
    return mouth_aspect_ratio


# ---------------------------------------------------------------------


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for idx, face in enumerate(faces):
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        x3 = (x1 + x2) / 2
        y3 = (y1 + y2) / 2

        face_crop = np.copy(frame[y1:y2, x1:x2])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}".format(label, conf[idx] * 100)
        if count > 12:
            if label == "man" and gender == True:
                pyautogui.press("M")
                gender = False
            elif label == "woman" and gender == True:
                pyautogui.press("F")
                gender = False

        count += 1

        Y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # write label above face rectangle
        cv2.putText(frame, label, (x1, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        if 400 > x3 > 300:
            x4 = 0
            pyautogui.keyUp("A")
            pyautogui.keyUp("D")
        elif x3 > 400:
            x4 = 1  # Left
            pyautogui.keyDown("A")
            pyautogui.keyUp("D")
        elif x3 < 300:
            x4 = -1  # Right
            pyautogui.keyDown("D")
            pyautogui.keyUp("A")

        if 250 > y3 > 200:
            y4 = 0
            pyautogui.keyUp("S")
            pyautogui.keyUp("W")
        elif y3 > 230:  # Down
            y4 = -1
            pyautogui.keyDown("S")
            pyautogui.keyUp("W")
        elif y3 < 200:  # Up
            y4 = 1
            pyautogui.keyDown("W")
            pyautogui.keyUp("S")

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        # mouth open detection part----------------------
        innerLip = []

        for n in range(61, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            innerLip.append((x, y))
            next_point = n + 1
            if n == 67:
                next_point = 61
            x2 = landmarks.part(next_point).x
            y2 = landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        inner_Lip = calculate_Distance(innerLip)
        MOUTH = inner_Lip
        MOUTH = round(MOUTH, 2)

        if MOUTH > 0.26:
            pyautogui.click()
        # --------------------------------------------------

        # Left Eye Open Detection------------------------------------
        leftEye = []

        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = landmarks.part(next_point).x
            y2 = landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_eye = calculate_Distance(leftEye)

        LEFT_EYE = left_eye
        LEFT_EYE = round(LEFT_EYE, 2)

        if LEFT_EYE < 0.20:
            pyautogui.press('1')
        # --------------------------------------------------

        # Right Eye Open Detection ---------------------------
        rightEye = []

        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = landmarks.part(next_point).x
            y2 = landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        right_eye = calculate_Distance(rightEye)

        RIGHT_EYE = right_eye
        RIGHT_EYE = round(RIGHT_EYE, 2)

        if RIGHT_EYE < 0.18:
            pyautogui.press("2")
            print(RIGHT_EYE)
    # -----------------------------------------------------

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
