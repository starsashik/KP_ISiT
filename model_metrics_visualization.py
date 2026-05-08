import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix

# Включаем логирование для отладки
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true, y_pred, save_path=None, classes=None,
                          figsize=(12, 12), cmap='Blues'):
    """ Вывод матрицы ошибок в виде тепловой карты. """

    if classes is None:
        classes = [
            "greeting",
            "goodbye",
            "thanks",
            "menu_request",
            "cart_request",
            "order_request",
            "price_request",
            "complete_order_request",
            "clear_cart_request",
            "failure_phrases"
        ]

    # Создаём матрицу ошибок
    cm = confusion_matrix(y_true, y_pred)

    # Создаём фигуру
    fig, ax = plt.subplots(figsize=figsize)

    # Рисуем тепловую карту
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

    # Настройки
    ax.set_xlabel('Предсказанные классы', fontsize=12, fontweight='bold')
    ax.set_ylabel('Истинные классы', fontsize=12, fontweight='bold')
    ax.set_title("Матрица ошибок", fontsize=14, fontweight='bold')

    # Поворот подписей для лучшей читаемости
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0, ha='right')

    plt.tight_layout()

    # Сохраняем если нужно
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"График сохранён в {save_path}")

    plt.show()

def plot_learning_curve(estimator, X, y, save_path=None, cv=5,
                        train_sizes=np.linspace(0.1, 1.0, 10),
                        scoring='accuracy', n_jobs=-1, figsize=(10, 6)):
    """ Вывод графика кривой обучения. """

    # Рассчитываем кривую обучения
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=42)

    # Вычисляем средние и стандартные отклонения
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Создаём фигуру
    fig, ax = plt.subplots(figsize=figsize)

    # Рисуем линии
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Обучающая выборка', linewidth=2)
    ax.plot(train_sizes_abs, test_mean, 'o-', color='red', label='Валидационная выборка', linewidth=2)

    # Добавляем доверительные интервалы
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='blue')
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                    alpha=0.15, color='red')

    # Настройки графика
    ax.set_xlabel('Размер обучающей выборки', fontsize=12, fontweight='bold')
    ax.set_ylabel(scoring.capitalize(), fontsize=12, fontweight='bold')
    ax.set_title("Кривая обучения", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Добавляем аннотации для лучшей читаемости
    ax.annotate(f'Лучшее качество: {test_mean.max():.2%}',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"График сохранён в {save_path}")

    plt.show()
