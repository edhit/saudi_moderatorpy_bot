import asyncio
import os
import logging
import asyncpg
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils import executor
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np
import config

# Настройки
BOT_TOKEN = config.BOT_TOKEN
GROUP_ID = config.GROUP_ID
REVIEWERS = config.REVIEWERS
DATABASE_URL = config.DATABASE_URL
MAX_FEATURES = config.MAX_FEATURES

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)

# Инициализация модели
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(MAX_FEATURES,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
model_loaded = False  # Флаг загрузки модели

# Токенизация текста
async def preprocess_text(text):
    """Токенизация текста."""
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)


async def get_training_data(conn):
    """Получение данных для обучения из базы данных."""
    rows = await conn.fetch("SELECT text, label FROM training_data")
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    return texts, np.array(labels)


async def save_training_data(conn, text, label):
    """Сохранение данных для обучения в базу данных."""
    await conn.execute("INSERT INTO training_data (text, label) VALUES ($1, $2)", text, label)


async def train_model():
    """Обучение модели в фоновом режиме."""
    global model, model_loaded
    async with asyncpg.connect(DATABASE_URL) as conn:
        texts, labels = await get_training_data(conn)
        if len(labels) < 1000:
            logging.info("Недостаточно данных для обучения.")
            return
        X = tfidf_vectorizer.fit_transform(texts).toarray()
        model.fit(X, labels, epochs=10, batch_size=32, verbose=1)
        model.save("moderation_model.h5")
        model_loaded = True
        logging.info("Модель успешно обучена.")


async def predict_message(text):
    """Предсказание для нового сообщения."""
    global model, model_loaded
    if not model_loaded and os.path.exists("moderation_model.h5"):
        model.load_weights("moderation_model.h5")
        model_loaded = True
    X = tfidf_vectorizer.transform([text]).toarray()
    prediction = model.predict(X)[0][0]
    return prediction > 0.5


@dp.message_handler(content_types=types.ContentType.TEXT, chat_type=types.ChatType.SUPERGROUP)
async def handle_group_message(message: types.Message):
    """Обработка сообщений из группы."""
    try:
        text = await preprocess_text(message.text)

        # Проверка через модель
        if os.path.exists("moderation_model.h5") and await predict_message(text):
            await message.delete()
            await message.reply("Ваше сообщение удалено: оно не соответствует правилам.")
            return

        # Рассылка модераторам
        markup = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton("Да", callback_data=f"approve_{message.message_id}_1"),
                InlineKeyboardButton("Нет", callback_data=f"approve_{message.message_id}_0")
            ]
        ])
        for reviewer in REVIEWERS:
            try:
                await bot.send_message(
                    reviewer,
                    f"Это сообщение подходит?\n\n{text}",
                    reply_markup=markup
                )
            except Exception as e:
                logging.warning(f"Ошибка отправки модератору {reviewer}: {e}")
    except Exception as e:
        logging.error(f"Ошибка обработки сообщения: {e}")


@dp.callback_query_handler(lambda c: c.data.startswith("approve_"))
async def handle_approval(callback_query: types.CallbackQuery):
    """Обработка ответа модераторов."""
    try:
        _, message_id, label = callback_query.data.split("_")
        label = int(label)

        message = await bot.get_message(chat_id=GROUP_ID, message_id=int(message_id))
        text = await preprocess_text(message.text)

        async with asyncpg.connect(DATABASE_URL) as conn:
            await save_training_data(conn, text, label)

        await callback_query.answer("Спасибо за ваш ответ!")
        asyncio.create_task(train_model())
    except Exception as e:
        logging.error(f"Ошибка обработки callback: {e}")


async def setup_database():
    """Создание таблицы в базе данных."""
    async with asyncpg.connect(DATABASE_URL) as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            label INT NOT NULL
        )
        """)
        logging.info("Таблица training_data проверена/создана.")


if __name__ == "__main__":
    asyncio.run(setup_database())
    executor.start_polling(dp, skip_updates=True)
