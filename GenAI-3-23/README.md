# Генератор вопросов по теме

Модуль генерирует пары вопрос-ответ по заданной теме или короткому тексту и сохраняет результат в CSV. Для генерации вопросов используется `google/flan-t5-large`, для поиска ответа используется `distilbert-base-uncased-distilled-squad`.

## Возможности

- Приём темы или короткого текста через аргумент командной строки.
- Генерация заданного количества пар вопрос-ответ.
- Сохранение результата в CSV-файл.
- Использование GPU при наличии.

## Установка

Все команды выполняются из каталога `GenAI-3-23`:

```bash
pip install -r requirements.txt
```

## Использование

```bash
python script.py "Artificial Intelligence and its applications" --output_file output.csv --num_questions 5
```

## Аргументы

| Аргумент | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `topic` | `str` | требуется | Тема или короткий текст для генерации вопросов и ответов. |
| `--output_file` | `str` | `output.csv` | Путь к выходному CSV-файлу. |
| `--num_questions` | `int` | `3` | Количество генерируемых пар вопрос-ответ. |
| `--max_tokens` | `int` | `256` | Максимальное количество токенов в сгенерированном вопросе. |

## Формат CSV

Файл содержит три колонки:

| Колонка | Описание |
| --- | --- |
| `sentence` | Контекст, на основе которого создан вопрос. |
| `question` | Сгенерированный вопрос. |
| `answer` | Ответ, найденный QA-моделью в контексте. |

## Пример

```bash
python script.py "Artificial Intelligence is used in healthcare, finance, and education." --num_questions 2
```

Пример результата:

| sentence | question | answer |
| --- | --- | --- |
| Artificial Intelligence is used in healthcare, finance, and education. | Where is Artificial Intelligence used? | healthcare, finance, and education |

## Примечания

- Рекомендуется использовать английский текст.
- При первом запуске модели будут загружены автоматически.
- На CPU генерация может выполняться медленно.
