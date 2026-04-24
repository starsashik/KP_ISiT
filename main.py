import asyncio
import random
import requests
from datetime import time

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from config import (
    TELEGRAM_TOKEN
)
from bot_logic import LunchMindBot


# Инициализация бота
bot = LunchMindBot()

# Обработчики Telegram
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для заказа обедов. Чем могу помочь?",
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Я могу помочь с меню, оформлением заказа и информацией о работе ресторана. Просто напишите!",
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        bot.show_menu(),
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def cart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)

    await update.message.reply_text(
        bot.show_cart(user_id),
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def clear_cart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)

    await update.message.reply_text(
        bot.clear_cart(user_id),
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )


async def complete_order_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)

    await update.message.reply_text(
        bot.complete_order(user_id),
        reply_markup=bot.menu_keyboard,
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_name = update.effective_user.username
    text = update.message.text

    if text == "📋 Меню":
        await menu_command(update, context)

        print(f"Пользователь ({user_id} - {user_name}): нажал кнопку \"меню\"")

    elif text == "🛒 Корзина":
        await cart_command(update, context)

        print(f"Пользователь ({user_id} - {user_name}): нажал кнопку \"корзина\"")

    elif text == "❌ Очистить корзину":
        await clear_cart_command(update, context)

        print(f"Пользователь ({user_id} - {user_name}): нажал кнопку \"очистить корзину\"")

    elif text == "✅ Оформить заказ":
        await complete_order_command(update, context)

        print(f"Пользователь ({user_id} - {user_name}): нажал кнопку \"оформить заказ\"")

    else:
        response = bot.handle_message(text, user_id)

        await update.message.reply_text(
            response,
            reply_markup=bot.menu_keyboard,
            parse_mode='Markdown'
        )

        print(f"Сообщение пользователя ({user_id} - {user_name}): {text}")
        print(f"Ответ бота: {response}")
        print()


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("cart", cart_command))
    app.add_handler(CommandHandler("clear_cart", clear_cart_command))
    app.add_handler(CommandHandler("complete_order", complete_order_command))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Бот запущен и ожидает сообщений пользователей...")

    app.run_polling()

if __name__ == "__main__":
    main()
