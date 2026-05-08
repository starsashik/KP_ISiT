import json
import re # инструменты для работы с регулярными выражениями

from spellchecker import SpellChecker # Для базовой проверки орфографии
import pymorphy2 # Для морфологического анализа и склонения слов
# Natasha решает базовые задачи обработки естественного языка на русском языке
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, Doc
# Yargy использует правила и словари для извлечения структурированной информации из русских текстов.
from yargy import Parser
from yargy.pipelines import morph_pipeline
from yargy.interpretation import fact

from config import MENU_FILE_PATH

#Инициализация инструментов NLP
segmenter = Segmenter() # Natasha
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

morph_analyzer = pymorphy2.MorphAnalyzer() # pymorphy
spell_checker = SpellChecker(language='ru') # spellchecker


def clean_text(text):
    """Очистка текста от лишних символов и приведение к нижнему регистру"""
    if not text:
        return ""
    stripped_text = text.lower().strip() # Приводим в нижний регистр и очищаем по краям
    symbol_filtered_text = re.sub(r'[^а-яёa-z0-9-\s]', '', stripped_text)
    space_filtered_text = re.sub(r'\s+', ' ', symbol_filtered_text)
    return space_filtered_text.strip()

def correct_text(text):
    """Проверка орфографии в тексте"""
    if not text:
        return ""
    words = text.split()
    corrected_words = [spell_checker.correction(word) if word in spell_checker else word for word in words]
    return ' '.join(corrected_words)

def lemmatize_text(text):
    """Лемматизация текста"""
    if not text:
        return ""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return ' '.join([token.lemma for token in doc.tokens if token.lemma])

def extract_entities(text):
    """Извлечение сущностей без дублирования"""
    if not text:
        return []

    entities = []
    seen = set()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    # NER сущности
    for span in doc.spans:
        span.normalize(morph_vocab)
        entity_type = span.type
        entity_text = span.text
        entity_normal = span.normal or entity_text

        key = (entity_type, entity_normal)
        if key not in seen:
            seen.add(key)
            entities.append({
                'type': entity_type,
                'text': entity_text,
                'normal': entity_normal
            })

    # Загрузка меню
    try:
        with open(MENU_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        menu_dishes = list(data.keys())
    except FileNotFoundError:
        menu_dishes = []
    # Создание парсера для блюд
    if menu_dishes:
        menu_item = fact('MenuItem', ['name'])
        menu_rule = morph_pipeline(menu_dishes).interpretation(menu_item.name).interpretation(menu_item)
        menu_parser = Parser(menu_rule)
    else:
        menu_parser = None
    # Блюда из меню (уникальные)
    if menu_parser is not None:
        for match in menu_parser.findall(text):
            dish_name = match.fact.name
            key = ('MENU_ITEM', dish_name.lower())
            if key not in seen:
                seen.add(key)
                entities.append({
                    'type': 'MENU_ITEM',
                    'text': dish_name,
                    'normal': dish_name.lower()
                })

    return entities

def   analyze_sentiment(text, emo_dict):
    """Анализ тональности текста"""
    lemmatized = lemmatize_text(text)
    words = lemmatized.split()

    sentiment_score = 0
    matched_words = 0

    for word in words:
        if word in emo_dict:
            sentiment_score += emo_dict[word]
            matched_words += 1

    if matched_words > 0:
        return sentiment_score / matched_words
    return 0

def lemmatize_correct_clean_text(text):
    return lemmatize_text(correct_text(clean_text(text)))