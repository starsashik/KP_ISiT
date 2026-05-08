from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import tempfile
from pathlib import Path
from pydub import AudioSegment
import speech_recognition as sr
from gtts import gTTS
import os
from pathlib import Path

from config import (
    TELEGRAM_TOKEN
)
from bot_logic import LunchMindBot

# Включаем логирование для отладки
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = LunchMindBot()
logger.info("Бот запущен")


# Обработчики Telegram
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Стандартная функция для запуска бота"""
    await update.message.reply_text(
        "Привет! Я бот для заказа обедов из ресторана. Чем могу помочь?",
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Вспомогательная функция для вывода сообщения с помощью"""
    await update.message.reply_text(
        "Я могу помочь с меню, оформлением заказа, информацией о работе ресторана и просто поддержать беседу. Просто напишите!",
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Функция обращается к боту вывода меню"""
    await update.message.reply_text(
        bot.show_menu(),
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def cart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Функция обращается к боту для вывода корзины"""
    user_id = str(update.message.from_user.id)
    await update.message.reply_text(
        bot.show_cart(user_id),
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def clear_cart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Функция обращается к боту для очистки корзины"""
    user_id = str(update.message.from_user.id)
    await update.message.reply_text(
        bot.clear_cart(user_id),
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def complete_order_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Функция обращается к боту для совершения заказа"""
    user_id = str(update.message.from_user.id)
    await update.message.reply_text(
        bot.complete_order(user_id),
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def handle_message_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает текстовые сообщения."""
    user_id = str(update.message.from_user.id)
    user_name = update.effective_user.username
    text = update.message.text

    if text == "🍽️ Меню":
        await menu_command(update, context)
        logger.info(f"Пользователь ({user_id} - {user_name}): нажал кнопку \"меню\"")

    elif text == "🛒 Корзина":
        await cart_command(update, context)
        logger.info(f"Пользователь ({user_id} - {user_name}): нажал кнопку \"корзина\"")

    elif text == "❌ Очистить корзину":
        await clear_cart_command(update, context)
        logger.info(f"Пользователь ({user_id} - {user_name}): нажал кнопку \"очистить корзину\"")

    elif text == "✔️ Оформить заказ":
        await complete_order_command(update, context)
        logger.info(f"Пользователь ({user_id} - {user_name}): нажал кнопку \"оформить заказ\"")

    else:
        response = bot.handle_message(text, user_id)

        await update.message.reply_text(
            response,
            reply_markup=bot.menu_keyboard,
            parse_mode='Markdown'
        )

        logger.info(f"Сообщение от пользователя ({user_id} - {user_name}): {text}")
        logger.info(f"Ответ бота: {response}\n")

        coupon_message = bot.coupon_message(user_id)

        # Если настроение пользователя слишком плохое, то мы извиняемся после нескольких таких сообщений
        if bot.get_user_sentiment(user_id) <= -0.75:
            sorry_message = bot.apologize(user_id)
            if sorry_message is not None:
                await update.message.reply_text(
                    sorry_message,
                    reply_markup=bot.menu_keyboard,
                    parse_mode='Markdown'
                )
        # Выдаем купон на скидку за каждые 5 покупок
        elif coupon_message:
            await update.message.reply_text(
                coupon_message,
                reply_markup=bot.menu_keyboard,
                parse_mode='Markdown'
            )


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает голосовые сообщения"""
    user_id = str(update.message.from_user.id)
    user_name = update.effective_user.username

    processing_message = await update.message.reply_text("🎧 Получил ваше голосовое сообщение. Обрабатываю аудио...")

    # Получаем объект голосового сообщения
    voice = update.message.voice
    # Получаем файл с серверов Telegram
    file = await context.bot.get_file(voice.file_id)

    # Создаем временную директорию для работы с файлами
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Пути для временных файлов
        ogg_file_path = tmp_path / f"{voice.file_unique_id}.ogg"
        wav_file_path = tmp_path / f"{voice.file_unique_id}.wav"

        try:
            # Скачиваем файл в формате .ogg
            await file.download_to_drive(ogg_file_path)
            logger.info(f"Файл {voice.file_unique_id}.ogg успешно скачан.")

            #  Конвертируем .ogg в .wav с помощью pydub
            audio_segment = AudioSegment.from_ogg(ogg_file_path)
            # Экспортируем в несжатый .wav для библиотеки распознавания
            audio_segment.export(wav_file_path, format="wav")
            logger.info(f"Файл успешно конвертирован в {wav_file_path.name}.")

            # Распознаем речь из .wav файла
            recognizer = sr.Recognizer()

            await processing_message.edit_text("🔄 Преобразую голос в текст...")

            with sr.AudioFile(str(wav_file_path)) as source:
                # Добавим небольшую паузу для обработки "шума" в начале записи
                recognizer.adjust_for_ambient_noise(source, duration=0.25)
                audio_data = recognizer.record(source)

            # Пытаемся распознать русскую речь с помощью Google Web Speech API (бесплатно)
            try:
                recognized_text = recognizer.recognize_google(audio_data, language="ru-RU")
                logger.info(f"Распознанный текст: {recognized_text}")

                # Отправляем результат пользователю
                await update.message.reply_text(
                    f"📝 **Распознанный текст:**\n\n"
                    f"_{recognized_text}_",
                    parse_mode="Markdown"
                )
                # Удаляем сообщение "Обрабатываю аудио..."
                await processing_message.delete()

                # Пытаемся обработать полученное сообщение
                response = bot.handle_message(recognized_text, user_id)

                # Отправляем ответ голосом!
                await text_to_voice_and_send(update, response)

                logger.info(f"Сообщение от пользователя ({user_id} - {user_name}): {response}")
                logger.info(f"Ответ бота: {response}\n")

                coupon_message = bot.coupon_message(user_id)

                # Если настроение пользователя слишком плохое, то мы извиняемся после нескольких таких сообщений
                if bot.get_user_sentiment(user_id) <= -0.75:
                    sorry_message = bot.apologize(user_id)
                    if sorry_message is not None:
                        await text_to_voice_and_send(update, sorry_message)
                # Выдаем купон на скидку за каждые 5 покупок
                elif coupon_message:
                    await text_to_voice_and_send(update, coupon_message)

            except sr.UnknownValueError:
                logger.warning("Не удалось распознать речь в сообщении.")
                await processing_message.edit_text(
                    "Извините, я не смог разобрать речь в вашем сообщении. Попробуйте записать его четче или отправьте текстом.")
            except sr.RequestError as e:
                logger.error(f"Ошибка при запросе к сервису распознавания: {e}")
                await processing_message.edit_text(
                    "Произошла техническая ошибка при обращении к серверу распознавания речи. Попробуйте чуть позже.")

        except Exception as e:
            logger.error(f"Общая ошибка при обработке голосового сообщения: {e}")
            await processing_message.edit_text(
                "Произошла непредвиденная ошибка при обработке вашего голосового сообщения.")

async def text_to_voice_and_send(update: Update, text: str):
    """Преобразует текст в голосовое сообщение и отправляет пользователю. """
    # Создаём временный файл
    temp_file = Path(f"temp_voice_{update.message.from_user.id}.ogg")

    try:
        # Конвертируем текст в речь
        tts = gTTS(text=text, lang='ru', slow=False)

        # gTTS создаёт MP3, его нужно сохранить
        mp3_file = temp_file.with_suffix('.mp3')
        tts.save(str(mp3_file))

        # Конвертируем MP3 в OGG
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(str(mp3_file))
        audio.export(str(temp_file), format="ogg")

        # Отправляем голосовое сообщение
        with open(temp_file, 'rb') as voice_file:
            await update.message.reply_voice(voice=voice_file)

        # Удаляем временные файлы
        if mp3_file.exists():
            mp3_file.unlink()
        if temp_file.exists():
            temp_file.unlink()

    except Exception as e:
        logger.error(f"Ошибка при создании голосового сообщения: {e}")
        # Если не получилось с голосом, отправляем текстом
        await update.message.reply_text(text)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Стандартные команды
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("cart", cart_command))
    app.add_handler(CommandHandler("clear_cart", clear_cart_command))
    app.add_handler(CommandHandler("complete_order", complete_order_command))

    # Обработчик текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message_text))
    # Обработчик голосовых сообщений
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    print("Бот запущен и ожидает сообщений пользователей...")

    app.run_polling()


if __name__ == "__main__":
    main()
