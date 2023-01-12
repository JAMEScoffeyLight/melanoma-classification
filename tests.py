from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import decode_predictions
from melanoma_classification import preprocess_image

def test_model_frog():
    img = Image.open("frog.jpg")
    # Предварительная обработка изображения
    x = preprocess_image(img)
    # Загружаем предварительно обученную модель
    model = EfficientNetB0(weights='imagenet')
    # Распознавание изображения
    preds = model.predict(x)
    classes = decode_predictions(preds, top=3)[0]
    frogs = {
        'tailed_frog': 0.9253198,
        'tree_frog': 0.023592785,
        'bullfrog': 0.0068289405
    }
    for cl in classes:
        assert isclose(cl[2], frogs[cl[1]])
        
def test_model_car():
    img = Image.open("car.jpg")
    # Предварительная обработка изображения
    x = preprocess_image(img)
    # Загружаем предварительно обученную модель
    model = EfficientNetB0(weights='imagenet')
    # Распознавание изображения
    preds = model.predict(x)
    classes = decode_predictions(preds, top=3)[0]
    cars = {
        'sports_car': 0.85632396,
        'racer': 0.05533105,
        'convertible': 0.021732405
    }
    for cl in classes:
        assert isclose(cl[2], cars[cl[1]])
    

    
def isclose(a, b, rel_tol=1e-05, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

test_model_frog()
test_model_car()
