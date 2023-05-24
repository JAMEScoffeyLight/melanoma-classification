import io
import os
import uuid
import streamlit as st
from PIL import Image
import numpy as np
from predict import prediction_func
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import pathlib

#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath


@st.cache(allow_output_mutation=True)
def load_model():
    return EfficientNetB0(weights='imagenet')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    uploadedfile = st.file_uploader(label='Выберите изображение для распознавания')
    if uploadedfile is not None:
        imagedata = uploadedfile.getvalue()
        st.image(imagedata)
        img = Image.open(io.BytesIO(imagedata))
        # Генерируем уникальный идентификатор для названия файла
        filename = str(uuid.uuid4()) + '.png'
        # Сохраняем изображение в папку input с уникальным именем
        if not os.path.exists(pathlib.Path(pathlib.Path.cwd(), 'input', 'siic-isic-224x224-images')):
            os.mkdir(pathlib.Path(pathlib.Path.cwd(), 'input', 'siic-isic-224x224-images'))
        img.save(pathlib.Path(pathlib.Path.cwd(), 'input', 'siic-isic-224x224-images', filename))
        return img, filename
    else:
        return None, None

# Загружаем предварительно обученную модель
# Тут обучаем модель
# Выводим заголовок страницы
st.title('Классификация изображений')
# Выводим форму загрузки изображения и получаем изображение
img = load_image()
# Показывам кнопку для запуска распознавания изображения
result = st.button('Распознать изображение')
# Если кнопка нажата, то запускаем распознавание изображения
if result:
    # Предварительная обработка изображения
    x = preprocess_image(img[0])
    # Распознавание изображения
    filePath = os.path.abspath(pathlib.Path(pathlib.Path.cwd(), 'input', 'siic-isic-224x224-images', img[1]))
    st.write('Path: ', filePath)
    weighPath = os.path.abspath('melanoma_detector_fold5.pkl')
    
    # Выводим заголовок результатов распознавания жирным шрифтом
    # используя форматирование Markdown
    st.write('Результаты распознавания:')
    # Выводим результаты распознавания
    preds = prediction_func(img[1], weighPath, filePath)
    st.dataframe({ 'Шанс, что это меланома': [preds[0][0] * 100], 'Шанс, что это не меланома': [preds[0][1] * 100] })
