import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tkinter import *
import threading
import time
from playsound import playsound
global count
count = 0


def play_sound_thread():
    global count
    playsound('C://Users//User//PycharmProjects//drowsinessdetection//yolov5//sound.mp3', block=True)
    count = 0
def openCV(x):
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='C://Users//User//PycharmProjects//drowsinessdetection//yolov5//best.pt',
                           force_reload=True)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        ret, frame = cap.read()
        results = model(frame)
        labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

        if (16 in labels):
            global count
            count = count + 1
            if count>5:
                for i in range(1):
                    thread = threading.Thread(target=play_sound_thread)
                    thread.start()

        cv2.imshow('YOLO', np.squeeze(results.render()))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    window = Tk()
    window.title("Drowsiness Detection")
    window.geometry("500x500+500+100")
    btn=Button(window, text="Click to start prediction", fg='blue')
    btn.bind('<Button-1>', openCV)
    btn.place(x=80, y=100)
    window.mainloop()