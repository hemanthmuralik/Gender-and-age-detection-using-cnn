import os, cv2, numpy as np, random

try:
    from tensorflow.keras.models import load_model
    TF=True
except:
    TF=False

EMO='emotion_model.h5'
AGE='age_prediction_model.h5'
EMOS=["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
AGES=["0-10","11-20","21-30","31-40","41-50","51+"]

def synth(i,w=480,h=360):
    img=np.full((h,w,3),240,np.uint8)
    x1,y1,x2,y2=w//2-80,h//2-110,w//2+80,h//2+110
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    emo=EMOS[(i//10)%len(EMOS)]
    age=AGES[(i//15)%len(AGES)]
    label=f"{emo}|{age}"
    (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
    cv2.rectangle(img,(x1,y1-30),(x1+tw,y1),(0,255,0),-1)
    cv2.putText(img,label,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
    return img

def run_synth():
    i=0
    while True:
        frame=synth(i,640,480)
        cv2.imshow("Synthetic Demo",frame)
        if cv2.waitKey(50)&0xFF==ord('q'):break
        i+=1

def run_real():
    emo=load_model(EMO)
    age=load_model(AGE)
    face=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    cap=cv2.VideoCapture(0)
    while True:
        ok,f=cap.read()
        if not ok:break
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        faces=face.detectMultiScale(g,1.3,5)
        for (x,y,w,h) in faces:
            roi=cv2.resize(g[y:y+h,x:x+w],(48,48))
            roi=roi.astype("float32")/255.0
            roi=np.expand_dims(roi,axis=(0,-1))
            e=emo.predict(roi)
            a=age.predict(roi)
            e_=int(np.argmax(e))
            a_=int(np.argmax(a))
            lbl=f"{EMOS[e_]}|{AGES[a_]}"
            cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),2)
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            cv2.rectangle(f,(x,y-30),(x+tw,y),(0,255,0),-1)
            cv2.putText(f,lbl,(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
        cv2.imshow("Real Demo",f)
        if cv2.waitKey(1)&0xFF==ord('q'):break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    if TF and os.path.exists(EMO) and os.path.exists(AGE):
        print("Real models found — running real demo")
        run_real()
    else:
        print("Models missing — running synthetic demo")
        run_synth()
