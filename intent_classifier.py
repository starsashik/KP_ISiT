import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_preparation_func import get_menu, get_intent_dataset
from model_metrics_visualization import plot_confusion_matrix, plot_learning_curve
from config import MENU_FILE_PATH

# Включаем логирование для отладки
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SAVE_PATH = "data/graphics/"

class IntentClassifier:
    """Классификатор намерений для ресторанного бота."""

    def __init__(self, analyzer='char', ngram_range=(3, 3)):
        """Инициализация классификатора с настраиваемыми параметрами. """
        self.analyzer = analyzer
        self.ngram_range = ngram_range

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram_range
            )),
            ("clf", LinearSVC())
        ])

        self.is_trained = False

    def train(self, X_train, X_test, y_train, y_test):
        """Обучение модели классификатора."""
        logger.info(f"Начало обучение классификатора")
        logger.info(f"   Количество примеров: {len(X_train)}")
        logger.info(f"   Количество классов: {len(set(y_train))}")
        logger.info(f"   Параметры: analyzer={self.analyzer}, ngram_range={self.ngram_range}")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        train_pred = self.pipeline.predict(X_train)
        val_pred = self.pipeline.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_test, val_pred)
        logger.info(f"\nРезультаты обучения:")
        logger.info(f"   Точность на обучении: {train_acc:.4f}")
        logger.info(f"   Точность на валидации: {val_acc:.4f}")
        logger.info(f"   Разница: {train_acc - val_acc:.4f}\n")

    def predict(self, text):
        """Предсказание намерения для текста."""
        if not self.is_trained:
            raise ValueError("Модель не обучена. Невозможно предсказать.")
        # Если text — это строка, оборачиваем в список
        if isinstance(text, str):
            return self.pipeline.predict([text])[0]
        else:
            return self.pipeline.predict(text)

    def save(self, file_path="models/intent_classifier.pkl"):
        """Сохранение модели в файл"""
        if not self.is_trained:
            raise ValueError("Модель не обучена. Невозможно сохранить.")
        with open(file_path, "wb") as file:
            pickle.dump(self.pipeline, file)
        logger.info(f"Модель сохранена в {file_path}")

    @classmethod
    def load(cls, file_path):
        """Загрузка модели из файла"""
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        classifier = cls()
        classifier.pipeline = data
        classifier.is_trained = True
        logger.info(f"Модель загружена из {file_path}")
        return classifier


def prepare_intents_dataset_for_model(file_path):
    """Подготовка датасета намерений для модели"""
    logger.info(f"Загрузка датасета из {file_path}...")
    data = get_intent_dataset(file_path, get_menu(MENU_FILE_PATH))

    X, y = [], []

    for intent, intent_data in data["intents"].items():
        for example in intent_data["examples"]:
            X.append(example)
            y.append(intent)

    logger.info(f"Загружено {len(X)} примеров, {len(set(y))} классов")

    classes = []
    for cls in y:
        if cls not in classes:
            classes.append(cls)

    return X, y, classes


def train_and_save_model(dataset_file_path, model_file_path, test_size=0.2,
                         analyzer='char', ngram_range=(3, 3)):
    """Обучение и сохранение модели, вывод метрик и графиков."""

    # Подготовка данных
    X, y, classes = prepare_intents_dataset_for_model(dataset_file_path)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=42
    )

    logger.info(f"\nРазделение данных:")
    logger.info(f"  Обучающая выборка: {len(X_train)}")
    logger.info(f"  Тестовая выборка: {len(X_test)}")

    # Создание и обучение классификатора
    classifier = IntentClassifier(
        analyzer=analyzer,
        ngram_range=ngram_range
    )

    # Обучение модели
    classifier.train(X_train, X_test, y_train, y_test)
    # Сохранение модели
    classifier.save(model_file_path)

    # Визуализация матрицы ошибок
    logger.info("\nПостроение матрицы ошибок. \nБудет выведена на графике.")
    y_pred = classifier.pipeline.predict(X_test)
    plot_confusion_matrix(
        y_test, y_pred,
        classes=classes,
        save_path= SAVE_PATH + "confusion_matrix.png"
    )

    # Кривая обучения
    logger.info("\nПостроение кривой обучения. \nБудет выведена на графике.")
    plot_learning_curve(
        classifier.pipeline,
        X, y,
        save_path=  SAVE_PATH + "learning_curve.png"
    )

    # Точность
    accuracy = np.mean(y_pred == y_test)
    logger.info(f"Точность : {accuracy:.4f}\n")
    # Отчет по классам
    logger.info("Отчет по классам:")
    logger.info(f"{classification_report(y_test, y_pred)}\n")

    return classifier
