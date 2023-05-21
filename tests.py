import os
from PIL import Image
from predict import prediction_func
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import decode_predictions
from main import preprocess_image

def test_model_frog():
    img = Image.open('toad.png')
    # Предварительная обработка изображения
    x = preprocess_image(img)
    filePath = os.path.abspath('toad.png')
    weighPath = os.path.abspath('melanoma_detector_fold5.pkl')
    # Распознавание изображения
    preds = prediction_func('toad.png', weighPath, filePath)
    actualResults = [0.0801, 0.9199]
    
    print(float('{:.4f}'.format(preds[0][0])), actualResults[0])
    assert isclose(float('{:.4f}'.format(preds[0][0])), actualResults[0])
    print(float('{:.4f}'.format(preds[0][1])), actualResults[1])
    assert isclose(float('{:.4f}'.format(preds[0][1])), actualResults[1])
        
def test_model_car():
    img = Image.open('car.png')
    # Предварительная обработка изображения
    x = preprocess_image(img)
    filePath = os.path.abspath('car.png')
    weighPath = os.path.abspath('melanoma_detector_fold5.pkl')
    # Распознавание изображения
    preds = prediction_func('car.png', weighPath, filePath)
    actualResults = [0.1188, 0.8812]
    
    print(float('{:.4f}'.format(preds[0][0])), actualResults[0])
    assert isclose(float('{:.4f}'.format(preds[0][0])), actualResults[0])
    print(float('{:.4f}'.format(preds[0][1])), actualResults[1])
    assert isclose(float('{:.4f}'.format(preds[0][1])), actualResults[1])
    

    
def isclose(a, b, rel_tol=1e-05, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

test_model_frog()
test_model_car()
