import json

from nlp_func import lemmatize_correct_clean_text

def get_emo_dict(file_path):
    """Считывание словаря kartaslovsent.csv для определения тональности текста"""
    emo_dict = dict()
    with open(file_path, "r", encoding="utf-8") as file:
        is_first_line = True
        for line in file.readlines():
            if is_first_line:
                is_first_line = False
                continue
            term, tag, value, pstv, ngtv, neut, dunno, pstvNgtvDisagreementRatio = line.strip().split(';')
            emo_dict[term] = float(value)
    return emo_dict

def get_menu(file_path):
    """Считывание данных из файла с меню"""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_dialogues(file_path):
    """Считывание датасета диалогов и приведение его в корректный формат"""
    with open(file_path, "r", encoding="utf8") as file:
        content = file.read()

    dialogues_str = content.split('\n\n')
    dialogues = [raw_dialogue.split('\n')[:2] for raw_dialogue in dialogues_str]

    dialogues_filtered = []
    questions = set()

    for dialogue in dialogues:
        if len(dialogue) != 2: # Проверка формата вопрос-ответ
            continue

        question, answer = dialogue
        # Очистка, проверка и лемматизация
        question = lemmatize_correct_clean_text(question[2:])
        answer = answer[2:]

        if question != '' and question not in questions:
            questions.add(question)
            dialogues_filtered.append([question, answer])

    dialogues_structured = {}

    for question, answer in dialogues_filtered:
        words = set(question.split(' '))
        for word in words:
            if word not in dialogues_structured:
                dialogues_structured[word] = []
            dialogues_structured[word].append([question, answer])

    structured_dialogues_cut = {}
    for word, pairs in dialogues_structured.items():
        pairs.sort(key=lambda pair: len(pair[0]))
        structured_dialogues_cut[word] = pairs[:1000]

    return structured_dialogues_cut

def get_intent_dataset(file_path, menu):
    """Считывание данных из файла с датасетом намерений, а также замена <DISH> на все возможные позиции из меню"""
    menu_dishes = list(menu.keys())

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for intent in list(data["intents"].keys()):
        examples = []

        for example in data["intents"][intent]["examples"]:
            if "<DISH>" in example:
                for dish in menu_dishes:
                    examples.append(example.replace("<DISH>", dish))
            else:
                examples.append(example)

        data["intents"][intent]["examples"] = examples

    return data