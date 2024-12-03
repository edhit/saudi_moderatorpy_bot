import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils import executor
import tensorflow as tf
import numpy as np
import pandas as pd

API_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
GROUP_ID = -123456789  # ID группы, откуда бот получает сообщения
reviewers = [111111111, 222222222]  # Telegram ID модераторов
training_data_file = "training_data.csv"
model_file = "moderation_model.h5"

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Нейросеть
input_dim = 100  # Размер входных данных
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def preprocess_text(text):
    """Препроцессинг текста: упрощение и векторизация."""
    # Простой пример векторизации текста
    return np.array([hash(word) % 1000 for word in text.split()[:input_dim]]).astype(float)


def load_training_data():
    """Загрузка обучающих данных из файла."""
    try:
        data = pd.read_csv(training_data_file)
        X = np.array([eval(row) for row in data['features']])
        y = data['label'].values
        return X, y
    except FileNotFoundError:
        return np.array([]), np.array([])


def save_training_data(features, label):
    """Сохранение данных для обучения."""
    new_entry = pd.DataFrame({'features': [features.tolist()], 'label': [label]})
    if not os.path.exists(training_data_file):
        new_entry.to_csv(training_data_file, index=False)
    else:
        new_entry.to_csv(training_data_file, mode='a', header=False, index=False)


def train_model():
    """Обучение модели."""
    X, y = load_training_data()
    if len(X) >= 1000:  # Тренировать только при достижении 1000 записей
        model.fit(X, y, epochs=10, verbose=1)
        model.save(model_file)


@dp.message_handler(content_types=types.ContentType.TEXT, chat_type=types.ChatType.SUPERGROUP)
async def handle_group_message(message: types.Message):
    """Обработка сообщений из группы."""
    text = message.text
    features = preprocess_text(text)

    if os.path.exists(model_file):
        model.load_weights(model_file)
        prediction = model.predict(np.array([features]))[0][0]
        if prediction > 0.5:  # Условие удаления сообщения
            await message.delete()
            await message.reply("Ваше сообщение было удалено. Оно не соответствует правилам.")
        return

    # Рассылка модераторам
    for reviewer in reviewers:
        try:
            markup = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton("Да", callback_data=f"approve_{message.message_id}_1"),
                 InlineKeyboardButton("Нет", callback_data=f"approve_{message.message_id}_0")]
            ])
            await bot.send_message(reviewer, f"Это сообщение подходит?\n\n{text}", reply_markup=markup)
        except Exception as e:
            continue


@dp.callback_query_handler(lambda c: c.data.startswith("approve_"))
async def handle_approval(callback_query: types.CallbackQuery):
    """Обработка ответа модераторов."""
    _, message_id, label = callback_query.data.split("_")
    label = int(label)

    # Получение текста сообщения
    message = await bot.get_message(chat_id=GROUP_ID, message_id=int(message_id))
    features = preprocess_text(message.text)

    # Сохранение данных для обучения
    save_training_data(features, label)

    await callback_query.answer("Спасибо за ваш ответ!")
    train_model()


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)