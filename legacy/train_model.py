# Reproducible baseline trainer for the supplied PlantVillage-style dataset.
# Uses color images only (part_1/color) to avoid mixing transformed variants.
import cv2, os, json, joblib, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

DATA_ROOT=os.environ.get('DATA_ROOT','dataset/part_1/color')
OUT='models'; os.makedirs(OUT,exist_ok=True)
classes=sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT,d))]); mp={c:i for i,c in enumerate(classes)}
hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9); X=[]; y=[]
for c in classes:
    for fn in os.listdir(os.path.join(DATA_ROOT,c)):
        if fn.lower().endswith(('.jpg','.jpeg','.png')):
            im=cv2.imread(os.path.join(DATA_ROOT,c,fn))
            if im is None: continue
            im=cv2.resize(im,(64,64)); h=hog.compute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)).ravel()
            hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); hist=cv2.calcHist([hsv],[0,1],None,[16,16],[0,180,0,256]).ravel(); hist/=hist.sum()+1e-8
            X.append(np.r_[h,hist]); y.append(mp[c])
X=np.asarray(X,np.float32); y=np.asarray(y)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=SGDClassifier(loss='log_loss',max_iter=50,tol=1e-3,random_state=42,n_jobs=-1,early_stopping=True,validation_fraction=.1)
model.fit(Xtr,ytr); pred=model.predict(Xte); acc=float(accuracy_score(yte,pred))
joblib.dump({'model':model,'classes':classes,'feature':'HOG+HSV histogram','image_size':[64,64]},OUT+'/crop_disease_model.joblib',compress=3)
json.dump(classes,open(OUT+'/classes.json','w'),indent=2)
json.dump({'accuracy':acc,'samples':int(len(y)),'train_samples':int(len(ytr)),'test_samples':int(len(yte)),'classes':len(classes)},open(OUT+'/metrics.json','w'),indent=2)
print(f'Accuracy: {acc:.4f}'); print('Saved models/crop_disease_model.joblib')
