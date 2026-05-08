import nltk
import random
from telegram import ReplyKeyboardMarkup

from data_preparation_func import get_emo_dict, get_dialogues, get_menu, get_intent_dataset
from nlp_func import extract_entities, analyze_sentiment, lemmatize_correct_clean_text
from intent_classifier import IntentClassifier, train_and_save_model

from config import MODEL_FILE_PATH, INTENT_DATASET_FILE_PATH, MENU_FILE_PATH, DIALOGUES_FILE_PATH, EMO_DICT_FILE_PATH

# Загрузка данных
EMO_DICT = get_emo_dict(EMO_DICT_FILE_PATH)
DIALOGUES = get_dialogues(DIALOGUES_FILE_PATH)
MENU = get_menu(MENU_FILE_PATH)
INTENT_DATASET = get_intent_dataset(INTENT_DATASET_FILE_PATH, MENU)

# Загрузка модели классификатора
try:
    INTENT_CLASSIFIER = IntentClassifier.load(MODEL_FILE_PATH)
except FileNotFoundError:
    print("Модель не найдена, обучаю модель заново")
    INTENT_CLASSIFIER = train_and_save_model(INTENT_DATASET_FILE_PATH, MODEL_FILE_PATH)


def _calculate_average_order_statistics(cart):
    """Вычисление средних параметров заказа"""
    if not cart:
        return None

    stats = {
        'spiciness': 0.0,
        'vegetarian': 0.0,
        'saltiness': 0.0,
        'sweetness': 0.0,
        'count': len(cart),
        'categories': dict()  # Добавляем подсчёт категорий
    }

    for item in cart:
        stats['spiciness'] += item.get('spiciness', 0.0)
        stats['vegetarian'] += item.get('vegetarian', 0.0)
        stats['saltiness'] += item.get('saltiness', 0.0)
        stats['sweetness'] += item.get('sweetness', 0.0)

        # Подсчёт категорий
        category = item.get('category', '')
        if category:
            stats['categories'][category] += 1

    stats['spiciness'] /= stats['count']
    stats['vegetarian'] /= stats['count']
    stats['saltiness'] /= stats['count']
    stats['sweetness'] /= stats['count']

    return stats


def _find_recommendation(cart):
    """Поиск блюда для рекомендации на основе текущей корзины"""
    if len(cart) == 0:
        return None

    order_stats = _calculate_average_order_statistics(cart)
    if order_stats is None:
        return None

    best_match = None
    best_score = -1

    # Определяем, что уже есть в заказе
    ordered_names = {item.get('name_lower', "") for item in cart}
    ordered_categories = set(item.get('category', '') for item in cart)

    for dish_name, dish_data in MENU.items():
        # Пропускаем уже заказанные блюда
        if dish_data.get('name_lower') in ordered_names:
            continue

        score = 0.0

        # Базовое соответствие вкусовых параметров
        score += 1.0 - abs(dish_data['spiciness'] - order_stats['spiciness'])
        score += 1.0 - abs(dish_data['saltiness'] - order_stats['saltiness'])
        score += 1.0 - abs(dish_data['sweetness'] - order_stats['sweetness'])

        # Вегетарианство
        vegetarian_preference = order_stats['vegetarian'] > 0.5
        is_vegetarian = dish_data['vegetarian'] > 0.5
        if vegetarian_preference and not is_vegetarian:
            score -= 0.8  # Сильный штраф для вегетарианца
        elif not vegetarian_preference and is_vegetarian:
            score -= 0.1  # Малый штраф для не-вегетарианца

        # Категориям блюд
        dish_category = dish_data['category']
        # Не рекомендовать два блюда из одной категории (кроме закусок и напитков)
        if dish_category not in ['закуски', 'напитки']:
            if dish_category in ordered_categories:
                score -= 0.3

        # Десерты
        is_dessert = dish_category == 'десерты'
        has_dessert = 'десерты' in ordered_categories
        if is_dessert:
            if has_dessert:
                score -= 0.5  # Уже есть десерт
            elif order_stats['sweetness'] < 0.3:
                score += 0.4  # Нет сладкого блюда - бонус к десерту
            elif order_stats['sweetness'] > 0.5:
                score -= 0.2  # Уже сладко, десерт не нужен
        # Штраф за основное блюдо после десерта
        if order_stats['sweetness'] > 0.5 and dish_data['sweetness'] < 0.3:
            score -= 0.2
        # Бонус за десерт после основного блюда
        if order_stats['sweetness'] < 0.3 and dish_data['sweetness'] > 0.7:
            score += 0.2

        # Напитки
        is_drink = dish_category == 'напитки'
        has_drink = 'напитки' in ordered_categories
        if is_drink:
            if has_drink:
                score -= 0.4  # Уже есть напиток
            else:
                score += 0.2  # Нет напитка - бонус
            if order_stats['spiciness'] > 0.5:
                score += 0.3  # К острому блюду напиток очень кстати

        # Основные блюда
        if dish_category in ['горячее', 'бургеры']:
            # Проверяем, есть ли уже основное блюдо
            has_main = any(cat in ['горячее', 'бургеры'] for cat in ordered_categories)
            if has_main:
                score -= 0.25  # Штраф за второе основное блюдо
            else:
                score += 0.1  # Бонус, если нет основного
                # Бонус, если есть салат/суп, но нет основного
                if 'салаты' in ordered_categories or 'супы' in ordered_categories:
                    score += 0.15

        # Салаты к основному
        if dish_category == 'салаты':
            has_main = any(cat in ['горячее', 'бургеры'] for cat in ordered_categories)
            if has_main:
                score += 0.2  # Салат хорошо идёт с основным
            else:
                score -= 0.1

        # Выбор лучшего
        if score > best_score:
            best_score = score
            best_match = dish_data

    return best_match


class LunchMindBot:
    def __init__(self):
        self.context = {}
        self.carts = {}

        self.menu_keyboard = ReplyKeyboardMarkup(
            [
                ["🛒 Корзина", "🍽️ Меню"],
                ["❌ Очистить корзину", "✔️ Оформить заказ"]
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
        """Вывод текста меню"""
        menu_text = "🍽️ *Наше меню*:\n\n"
        for item, details in MENU.items():
            menu_text += f"*{details['name']}*:\n"
            menu_text += f"{details['description']}\n"
            menu_text += f"Цена: {details['price']} руб.\n\n"
        menu_text += "Если хотите что-то заказать, то напишите об этом"

        return menu_text

    def show_cart(self, user_id):
        """Генерация текста содержимого корзины"""
        cart = self._get_user_cart(user_id)

        if len(cart) == 0:
            return "Ваша корзина пуста."

        total = 0

        # Создание красивого ответа
        cart_text = "🛒 *Ваша корзина*:\n\n"
        for item in cart:
            cart_text += f"- {item['name']} - {item['price']} руб.\n"
            total += item["price"]
        cart_text += f"\n*Итого: {total} руб.*"

        return cart_text

    def clear_cart(self, user_id):
        """Очистка корзины"""
        self.carts[user_id] = []
        return "❌ *Корзина очищена*"

    def complete_order(self, user_id):
        """Оформление заказа"""
        cart = self._get_user_cart(user_id)

        if len(cart) == 0:
            return "Ваша корзина пуста. Добавьте что-нибудь из меню."

        total = 0

        # Создание красивового ответа
        order_text = "✔️ *Ваш заказ оформлен!*\n\n"
        for item in cart:
            order_text += f"- {item['name']} - {item['price']} руб.\n"
            total += item["price"]
        order_text += f"\n*Итого: {total} руб.*\n\n"
        order_text += "Спасибо за заказ! Ожидайте подтверждения."

        # Очищаем корзину так как уже сделали заказ
        self.clear_cart(user_id)

        if self.context[user_id]["sentiment"] > 0.4 and self.context[user_id]["recommendation_counter"] > 5:
            recommendation = _find_recommendation(cart)

            if recommendation is not None:
                order_text += (
                    "\n\nПроанализировав ваш заказ, мы подготовили персональную рекомендацию и думаем это может вам понравиться. "
                    "Вы можете заказать это прямо сейчас или при следующем визите!\n\n")
                order_text += "*Рекомендуем попробовать*:\n"
                order_text += f"{recommendation['name']} - {recommendation['price']} руб.\n"
                order_text += recommendation['description']

                self.context[user_id]["recommendation_counter"] = 0

        if self.context[user_id]["sentiment"] >= 0:
            self.context[user_id]["coupon_counter"] += 1

        return order_text

    def handle_message(self, text, user_id):
        """Обработка сообщения"""

        # Создание записи о пользователе, если до этого не было
        if user_id not in self.context:
            self.context[user_id] = {
                "last_intent": None,
                "sentiment": 0,
                "recommendation_counter": 0,
                "coupon_counter": 0,
                "apologize_counter": 0
            }

        prepared_text = lemmatize_correct_clean_text(text)

        sentiment = analyze_sentiment(prepared_text, EMO_DICT)
        entities = extract_entities(prepared_text)
        potential_intent = INTENT_CLASSIFIER.predict(
            prepared_text)  # нахождение потенциального интента при помощи модели

        intent = None
        for example in INTENT_DATASET["intents"][potential_intent]["examples"]:
            prepared_example = lemmatize_correct_clean_text(example)
            distance = nltk.edit_distance(prepared_text, prepared_example)
            if prepared_example and distance / len(prepared_example) <= 0.3:
                intent = potential_intent
                break

        # Подсчет нового настроения
        new_user_sentiment = (self.context[user_id]["sentiment"] + sentiment) / 2
        if new_user_sentiment > 1:
            new_user_sentiment = 1
        elif new_user_sentiment < -1:
            new_user_sentiment = -1

        # Инкремент счетчика для рекомендации
        new_user_recommendation_counter = self.context[user_id]["recommendation_counter"] + 1

        # Счетчики для купона и извинений оставляем без изменения
        old_user_coupon_counter = self.context[user_id]["coupon_counter"]
        old_user_apologize_counter = self.context[user_id]["apologize_counter"]

        self.context[user_id] = {
            "last_intent": None,
            "sentiment": new_user_sentiment,
            "recommendation_counter": new_user_recommendation_counter,
            "coupon_counter": old_user_coupon_counter,
            "apologize_counter": old_user_apologize_counter
        }

        if intent is not None:
            self.context[user_id]["last_intent"] = intent

            if intent == "greeting":
                return self._handle_greeting()
            elif intent == "goodbye":
                return self._handle_goodbye()
            elif intent == "menu_request":
                return self._handle_menu_request()
            elif intent == "cart_request":
                return self._handle_cart_request(user_id)
            elif intent == "order_request":
                return self._handle_order_request(entities, user_id)
            elif intent == "price_request":
                return self._handle_price_request(entities)
            elif intent == "complete_order_request":
                return self._handle_complete_order_request(user_id)
            elif intent == "clear_cart_request":
                return self._handle_clear_cart_request(user_id)

        else:
            response = self._generate_answer(text)
            if response is not None:
                return response

        return self._handle_failure_phrases()

    def _handle_greeting(self):
        """Обработка намерения приветствия"""
        responses = INTENT_DATASET["intents"]["greeting"]["responses"]
        return random.choice(responses)

    def _handle_goodbye(self):
        """Обработка намерения прощания"""
        responses = INTENT_DATASET["intents"]["goodbye"]["responses"]
        return random.choice(responses)

    def _handle_menu_request(self):
        """Обработка намерения получить меню"""
        responses = INTENT_DATASET["intents"]["menu_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        menu_text = self.show_menu()

        response += menu_text

        return response

    def _handle_cart_request(self, user_id):
        """Обработка намерения получить данные о корзине"""
        responses = INTENT_DATASET["intents"]["cart_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        self._get_user_cart(user_id)
        cart_text = self.show_cart(user_id)

        response += cart_text

        return response

    def _handle_order_request(self, entities, user_id):
        """Обработка намерения заказать блюдо"""
        responses = INTENT_DATASET["intents"]["order_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        # Поиск возможных блюд из сообщения пользователя
        possible_items = [entity["normal"] for entity in entities if entity["type"] == "MENU_ITEM"]
        prepared_possible_items = [lemmatize_correct_clean_text(item) for item in possible_items]

        # Проверка до первого найденного блюда
        ordered_item = None
        for item, item_data in MENU.items():
            if lemmatize_correct_clean_text(item_data["name"]) in prepared_possible_items:
                ordered_item = (item, item_data)
                break

        if ordered_item is not None:
            response += f"{ordered_item[1]['name']} - {ordered_item[1]['price']} руб.\n\n"

            self._get_user_cart(user_id)  # вызываем, чтобы избежать ошибки когда не создана корзина для пользователя
            self.carts[user_id].append(MENU[ordered_item[0]])  # Добавляем в корзину

            response += f"Товар добавлен в ваш заказ"

            return response

        return "Извините, я не понял, что вы хотите заказать. Можете уточнить или попробовать еще раз?"

    def _handle_price_request(self, entities):
        """Обработка намерения узнать цену блюда"""
        responses = INTENT_DATASET["intents"]["price_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        # Поиск возможных блюд из сообщения пользователя
        possible_items = [entity["normal"] for entity in entities if entity["type"] == "MENU_ITEM"]
        prepared_possible_items = [lemmatize_correct_clean_text(item) for item in possible_items]

        # Проверка до первого найденного блюда
        correct_item = None
        for item, item_data in MENU.items():
            if lemmatize_correct_clean_text(item_data["name"]) in prepared_possible_items:
                correct_item = (item, item_data)
                break

        if correct_item is not None:
            response += f"{correct_item[1]['name']} - {correct_item[1]['price']} руб.\n"
            response += f"{correct_item[1]['description']}\n\n"
            response += f"Если хотите заказать это - просто напишите об этом"

            return response

        return (f"Извините, я не понял, на какие блюда вы хотите узнать цену. "
                f"Попробуйте снова или можете посмотреть все меню целиком")

    def _handle_complete_order_request(self, user_id):
        """Обработка намерения оформить заказ"""
        responses = INTENT_DATASET["intents"]["complete_order_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        response += self.complete_order(user_id)

        return response

    def _handle_clear_cart_request(self, user_id):
        """Обработка намерения очистить корзину"""
        responses = INTENT_DATASET["intents"]["clear_cart_request"]["responses"]
        response = f"{random.choice(responses)}\n\n"

        self.clear_cart(user_id)

        return response

    def _generate_answer(self, text):
        """Генерация ответа на основе датасета диалогов"""
        prepared_text = lemmatize_correct_clean_text(text)
        words = set(prepared_text.split(" "))
        mini_dataset = []
        for word in words:
            if word in DIALOGUES:
                mini_dataset += DIALOGUES[word]
        mini_dataset = set(mini_dataset)

        answers = []
        for question, answer in mini_dataset:
            prepared_question = lemmatize_correct_clean_text(question)

            if abs(len(prepared_text) - len(prepared_question)) / len(prepared_question) < 0.2:
                distance = nltk.edit_distance(prepared_text, prepared_question)
                distance_weighted = distance / len(prepared_question)

                if distance_weighted < 0.2:
                    answers.append([distance_weighted, question, answer])

        if answers:
            return min(answers, key=lambda x: x[0])[2]
        return None

    def _handle_failure_phrases(self):
        """Обработка нераспознанного намерения"""
        responses = INTENT_DATASET["failure_phrases"]

        return random.choice(responses)

    def get_user_sentiment(self, user_id):
        """Получение настроения пользователя"""
        return self.context[user_id]["sentiment"]

    def apologize(self, user_id):
        """Извинение перед пользователем"""
        sorry_message = (f"Похоже, я вас немного расстроил, и мне искренне жаль, что не могу помочь...\n\n"
                         f"Пожалуйста, свяжитесь с нашим менеджером: restaurant@restaurant.com\n"
                         f"Туда же можно отправить отзыв о моей работе — я учусь на ошибках!\n\n"
                         f"А в качестве извинений у меня для вас есть купон \"SORRY10\" на скидку 10% "
                         f"в любом нашем ресторане. Покажите его официанту, и мы загладим вину 😊")
        if self.context[user_id]["apologize_counter"] < 3:
            self.context[user_id]["apologize_counter"] += 1
        else:
            self.context[user_id]["apologize_counter"] = 0
            return sorry_message
        return None

    def coupon_message(self, user_id):
        """Генерация купона при достижении лимита"""
        if self.context[user_id].get("coupon_counter", 0) >= 5:
            self.context[user_id]["coupon_counter"] = 0

            # Генерация кода
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            coupon_code = ''.join(random.choice(chars) for _ in range(8))

            return (f"У нас для вас подарок! Мы подготовили персональный купон на скидку в нашем ресторане. \n\n"
                    f"```{coupon_code}```\n\n"
                    f"Купон начнет действовать уже со следующего заказа, просто предъявите его официанту или введите при заказе!")
        return ""
