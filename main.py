import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from pyzbar.pyzbar import decode

app = Flask(_name_)

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y") 



def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Vtu_No']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

def start():
    with open('authorised_list3.txt', 'r+') as f:
        f.truncate(0)
    cap = cv2.VideoCapture(0)
    ret = True
    i=0
    while ret:
            ret, frame = cap.read()
            if extract_faces(frame) != ():
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,cv2.LINE_AA)
                i+=1
            if i==10:
                break
            cv2.imshow('Face Scan', frame)
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html')

def barScanner():
    video = cv2.VideoCapture(0)
    video.set(3, 640)
    video.set(4, 740)

    with open('authorised_list3.txt', 'r') as file:
        authorised_list = file.read().strip()
        # print(authorised_list)
    i=0
    while True:
        success, image = video.read()
        for barcode in decode(image):
            qr_text = barcode.data.decode('utf-8')
            qr_text = str(qr_text).lower()
            if qr_text not in authorised_list:
                color = (0, 0, 255)
                display_message = "Denied Access"
                print("Access Denied")


            else:
                color = (0, 255, 0)
                display_message = "Access Granted"
                print("Access Granted")

                with open('QR_Registered.csv', 'r+') as f:
                    myDataList = f.readlines()
                    nameList = []
                    for line in myDataList:
                        entry = line.split(',')
                        nameList.append(entry[0])
                    if qr_text not in nameList:
                        now = datetime.now()
                        dtString = now.strftime('%H:%M:%S')
                        dtString1 = date.today()
                        f.writelines(f'\n{qr_text},{dtString1},{dtString}')
                return render_template('profile.html')
            polygon_points = np.array([barcode.polygon], np.int32)
            polygon_points = polygon_points.reshape(-1, 1, 2)
            rect_points = barcode.rect
            cv2.polylines(image, [polygon_points], True, color, 3)
            cv2.putText(image, display_message, (rect_points[0], rect_points[1]), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 2)
            i+=1
        if i==1:
            break
        cv2.imshow("QR Code Scanner", image)
        if cv2.waitKey(1) == 27:
            break
    video.release()
    cv2.destroyAllWindows()
    return render_template('index.html')


def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()

    return render_template('index.html')




if _name_ == '_main_':
    app.run(debug=True)
