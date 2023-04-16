import io
import os
import uuid
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions


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
        filename = str(uuid.uuid4()) + '.jpg'
        # Сохраняем изображение в папку input с уникальным именем
        if not os.path.exists('input'):
            os.makedirs('input')
        img.save(os.path.join('input', filename))
        return img, filename
    else:
        return None, None

def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])


# Загружаем предварительно обученную модель
model = load_model()
# Выводим заголовок страницы
st.title('Классификация изображений')
# Выводим форму загрузки изображения и получаем изображение
img = load_image()
st.write('Название изображения: ', img[1])
# Показывам кнопку для запуска распознавания изображения
result = st.button('Распознать изображение')
# Если кнопка нажата, то запускаем распознавание изображения
if result:
    # Предварительная обработка изображения
    x = preprocess_image(img[0])
    # Распознавание изображения
    preds = model.predict(x)
    # Выводим заголовок результатов распознавания жирным шрифтом
    # используя форматирование Markdown
    st.write('Результаты распознавания:')
    # Выводим результаты распознавания
    print_predictions(preds)