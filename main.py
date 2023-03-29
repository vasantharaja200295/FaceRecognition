import threading
import cv2

from deepface import DeepFace


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

counter = 0

match = False

reference = cv2.imread('vasantharaja.png')

def check_face(frame):
    global match
    try:
        if DeepFace.verify(frame,reference.copy())['verified']:
            match=True
    except ValueError:
        match = False



while True:
    ret, frame = cap.read()
    if ret:
        if counter%30==0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass

        counter+=1

        if match:
            cv2.putText(frame, "Match!",(20,450),cv2.FONT_HERSHEY_SIMPLEX,2, (2,255,0),3)
        else:
            cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        cv2.imshow("Video", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
