import streamlit as st
import torch
from PIL import Image

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import timm

# ---------------------------------------------------------------------
# Кэшируем загрузку модели, чтобы не перезагружать её при каждом обновлении страницы
@st.cache_resource
def load_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, processor, tokenizer


def generate_caption(image: Image.Image, model, processor, tokenizer) -> str:
    # Настройки генерации
    gen_kwargs = {
        "max_length": 16,
        "num_beams": 4,
        "no_repeat_ngram_size": 2,
    }

    # Преобразование изображения для модели
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Генерация подписи
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption


def main():
    st.title("Демо: Генерация описания к изображению")

    # Загрузка модели
    model, processor, tokenizer = load_model()

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите изображение:", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Открываем загруженное изображение
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        # Генерируем описание
        with st.spinner("Генерируем описание..."):
            caption = generate_caption(image, model, processor, tokenizer)

        st.subheader("Результат:")
        st.write(caption)


if __name__ == "__main__":
    main()
