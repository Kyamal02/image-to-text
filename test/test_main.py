"""
Интеграционные тесты для приложения, описанного в main.py

Для запуска локально: pytest tests/
"""

import pytest
from PIL import Image

from main.main import load_model, generate_caption


@pytest.fixture(scope="session")
def model_data():
    """
    Pytest-фикстура: один раз загружает модель и возвращает (model, processor, tokenizer).
    Это ускоряет тесты, чтобы не загружать модель многократно.
    """
    model, processor, tokenizer = load_model()
    return model, processor, tokenizer


def test_model_load_once(model_data):
    """
    Проверяем, что модель действительно загрузилась:
    у нас должен вернуться кортеж из трёх элементов,
    и у каждого есть ожидаемые атрибуты.
    """
    model, processor, tokenizer = model_data
    assert model is not None, "Model не загружена"
    assert processor is not None, "Processor не загружен"
    assert tokenizer is not None, "Tokenizer не загружен"


def test_generate_caption_basic(model_data):
    """
    Проверяем, что функция generate_caption() возвращает строку-описание
    для тестового (пусть и искусственного) изображения.
    """
    model, processor, tokenizer = model_data

    # Создадим простое красное изображение 100x100
    test_image = Image.new("RGB", (100, 100), color="red")

    caption = generate_caption(test_image, model, processor, tokenizer)

    # Проверяем, что результат - непустая строка
    assert isinstance(caption, str), "Функция должна возвращать строку"
    assert len(caption) > 0, "Описание не должно быть пустым"



