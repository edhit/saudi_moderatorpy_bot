# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . /app/

# Переменная окружения для настройки работы приложения
ENV PYTHONUNBUFFERED=1

# Открываем порт для работы с приложением
EXPOSE 8000

# Команда для запуска бота
CMD ["python", "bot.py"]
