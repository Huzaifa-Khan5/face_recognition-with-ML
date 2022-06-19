from itertools import count
from tkinter import *
import joblib
import numpy as np
import cv2
import json
from tkinter import filedialog
from Wavelet import w2d

with open('pehchan.pkl', 'rb') as f:
        model = joblib.load(f)
with open("class_dictionary.json", "r") as f:
    __class_name_to_number = json.load(f)
    __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')

def cameraProcessing():
    print("Streaming started")
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        cropped_faces=[]
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cropped_faces.append(roi_color) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result=[]
        result1=[]
        for img in cropped_faces:
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
            len_image_array = 32*32*3 + 32*32
            final = combined_img.reshape(1,len_image_array).astype(float)
            result.append(class_number_to_name(model.predict(final)[0]))
            result1.append(np.around(model.predict_proba(final)*100,2).tolist()[0])
            print(np.around(model.predict_proba(final)*100,2).tolist()[0])
            count=0
            if len(result1) >=2:
                for i in result1:
                    print(i)
                    k=((max(i)))
                    print(k)
                    if k<50:
                        result[count]="Unknown"  
                    count+=1 
            else:
                for i in result1:
                    j=max(i)
                    if j<50:
                        result[0]="Unknown"
        for ((x, y, w, h), name) in zip(faces, result):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0, 255), 2)
            cv2.putText(frame, name.title(), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()

def pictureProcessing():
    path=filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
    cropped_faces=[]
    img = cv2.imread(path)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cropped_faces.append(roi_color)
    result=[]
    result1=[]
    for img in cropped_faces:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append(class_number_to_name(model.predict(final)[0]))
        result1.append(np.around(model.predict_proba(final)*100,2).tolist()[0])
    count=0
    if len(result1) >=2:
        for i in result1:
            print(i)
            k=((max(i)))
            print(k)
            if k<66:
                result[count]="Unknown" 
            count+=1     
    for ((x, y, w, h),name) in zip(faces,result):
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0,255), 2)
        cv2.putText(rgb, name.title(), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)
    cv2.imshow("picture",rgb)
    cv2.waitKey()

root=Tk()
root.title("Pehchan")
root.geometry('700x600+270+50')
root.config(bg="#0c1220")

topframe=Frame(root,bg="#0c1220")
topframe.pack()

logo=PhotoImage(file="logo.png")
label=Label(topframe)
resizelogo=logo.subsample(1,1)
label.config(image=resizelogo)
label.pack()

middleframe=Frame(root,bg="#0c1220")
middleframe.pack()

label2=Label(middleframe,bg="#0c1220")
label2.pack(pady=5)

Endframe=Frame(root,bg="#0c1220")
Endframe.pack()

button=Button(Endframe,command=pictureProcessing,width=100,height=90,borderwidth=-10)
img_pic=PhotoImage(file="image.png")
resizeimage=img_pic.subsample(7,7)
button.config(image=resizeimage)
button.pack(side=LEFT,padx=100)

button2=Button(Endframe,command=cameraProcessing,width=100,height=90,borderwidth=-10)
camera_pic=PhotoImage(file="camera.png")
resizeimage2=camera_pic.subsample(6,5)
button2.config(image=resizeimage2)
button2.pack()

Endframe=Frame(root,bg="#0c1220")
Endframe.pack()

ilabel=Label(Endframe,text="By Image",font="BellMT",bg="#0c1220",fg="White")
ilabel.pack(side=LEFT,padx=110)

clabel=Label(Endframe,text="Real Time",font="BellMT",bg="#0c1220",fg="White")
clabel.pack(side=LEFT)

root.mainloop()