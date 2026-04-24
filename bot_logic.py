from telegram import ReplyKeyboardMarkup

class LunchMindBot:
    def __init__(self):
        self.context = {}
        self.carts = {}

        self.menu_keyboard = ReplyKeyboardMarkup(
            [
                ["🛒 Корзина", "📋 Меню"],
                ["❌ Очистить корзину", "✅ Оформить заказ"]
            ],
            resize_keyboard=True,
            one_time_keyboard=False
        )

    def _get_user_cart(self, user_id):
        """Получение корзины пользователя"""
        if user_id not in self.carts:
            self.carts[user_id] = []

        return self.carts[user_id]

    def show_menu(self):
        pass

    def show_cart(self, user_id):
        pass

    def clear_cart(self, user_id):
        pass

    def complete_order(self, user_id):
        pass

    def handle_message(self, text, user_id):
        pass

