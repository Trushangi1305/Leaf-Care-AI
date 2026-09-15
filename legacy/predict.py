import cv2, joblib, numpy as np

_bundle = joblib.load('models/crop_disease_model.joblib')
_model = _bundle['model']
_classes = _bundle['classes']
_hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)

def _features(path):
    im=cv2.imread(path)
    if im is None: raise ValueError('Unable to read image')
    im=cv2.resize(im,(64,64))
    h=_hog.compute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)).ravel()
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[16,16],[0,180,0,256]).ravel()
    hist=hist/(hist.sum()+1e-8)
    return np.asarray([np.r_[h,hist]],dtype=np.float32)

def predict(path):
    x=_features(path)
    idx=int(_model.predict(x)[0])
    result={'class':_classes[idx]}
    if hasattr(_model,'predict_proba'):
        p=_model.predict_proba(x)[0]
        result['confidence']=float(p[idx])
    return result

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2: raise SystemExit('Usage: python prediction_model/predict.py <image>')
    print(predict(sys.argv[1]))
