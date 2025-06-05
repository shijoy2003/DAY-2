import cv2
from deepface import Deepface
from spacy.lang.tokenizer_exceptions import emoticons
from srsly.ruamel_yaml import enforce

from main import results

cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()

    results = Deepface.analyze(img, actions=['emotions'], enforce_detection=False)

    emoticons = results[0]['dominant_emotions']
    cv2.putText(img, f'Emotions: {emoticons}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)

    cv2.imshow ("Emotion Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow()