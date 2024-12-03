# config.py
import os

BOT_TOKEN = os.getenv('BOT_TOKEN')
GROUP_ID = os.getenv('GROUP_ID')
REVIEWERS = [111111111, 222222222]  # Telegram ID модераторов
DATABASE_URL = os.getenv('DATABASE_URL')
MAX_FEATURES = os.getenv('MAX_FEATURES')  # Максимальное количество признаков TF-IDF