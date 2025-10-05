# Генератор вопросов по тексту
Проект позволяет генерировать вопросы по заданному тексту с использованием моделей **T5** для генерации вопросов и **DistilBERT QA** для генерации ответов. Сгенерированные вопросы и ответы, а также контекст, сохраняются в текстовый файл.

---

## Возможности

- Генерация нескольких вопросов по одному тексту.
- Поддержка GPU при наличии.
- Полная параметризация через аргументы командной строки.

---

## Требования

- Python 3.8+
- `torch`
- `transformers`
- `accelerate`
- `spacy`

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Установка `en_core_web_sm` для работы `spacy`:

```bash
python -m spacy download en_core_web_sm
```

---

## Использование

Запуск скрипта из командной строки:

```bash
python script.py data/text.txt --output_file data/output.txt --num_questions 5
```

### Аргументы командной строки

| Аргумент | Тип | По умолчанию           | Описание                                                  |
|----------|-----|------------------------|-----------------------------------------------------------|
| `--text_file` | str | требуется              | Путь входному файлу                                       |
| `--output_file` | str | `{input_file}_Q&A.txt` | Путь к выходному файлу                                    |
| `--num_questions` | int | 3                      | Количество генерируемых пар вопрос-ответ                  |
| `--max_tokens` | int | 256                    | Максимальное количество токенов в сгенерированном вопросе |

---

## Пример

Файл `data/text.txt`:

```
Text from data/text.txt
```

Команда для запуска:

```bash
python script.py --text_file data/text.txt
```

Вывод (в файл `data/text_Q&A.txt`):

```
S: Our good and wonderful sovereign has to perform the noblest role on earth, and he is so virtuous and noble that God will not forsake him.
Q: What is the main idea of the passage?
A: he is so virtuous and noble
--------------------------------------------------
S: "Can one be well while suffering morally?Can one be calm in times like these if one has any feeling?" said Anna Pavlovna.
Q: What did Anna Pavlovna say?
A: calm
--------------------------------------------------
S: Perhaps I don't understand things, but Austria never has wished, and does not wish, for war.
Q: What may be the reason Austria never has wished, and does not wish, for war?
A: Perhaps I don't understand things
```

---

## Как это работает

1. **Разбиение текста:** Используется spacy для сегментации предложений из текста.
2. **Генерация вопросов:** Используется модель T5 `valhalla/t5-base-qg-hl`.
3. **Генерация ответов:** Используется QA модель `distilbert-base-uncased-distilled-squad`.
4. **Вывод:** Релевантные предложения из текста, вопросы и ответы к ним сохраняются в файл.

---

## Примечания

- Рекомендуется использовать тексты на английском языке для оптимальной генерации.
- Если есть GPU, PyTorch автоматически его использует.
- Скрипт проверяет наличие файла и пустоту текста, ошибки обрабатываются и выводятся в консоль.

---


