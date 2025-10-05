import argparse
import os
import random
import re
from typing import List, Dict

import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def read_text_file(file_path: str) -> str:
    """Читает содержимое текстового файла"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    with open(file_path, "r", encoding="utf-8") as text_input:
        text = text_input.read()

    if not text.strip():
        raise ValueError("Ошибка: входной файл пустой.")

    return text


def write(results: List[Dict[str, str]], output_file: str = "output.txt") -> None:
    """Записывает список вопросов и ответов в текстовый файл"""
    try:
        with open(output_file, "w", encoding="utf-8") as text_output:
            for item in results:
                line = f"S: {item['sentence']}\nQ: {item['question']}\nA: {item['answer']}\n{'-' * 50}\n"
                text_output.write(line)
        print(f"Результаты записаны в {output_file}")
    except Exception as e:
        print(f"Ошибка при записи файла: {e}")


def preprocess_text(text: str) -> str:
    """Очистка текста от лишних пробелов и переносов строк"""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_into_sentences(text: str) -> List[str]:
    """Сегментация текста на предложения с использованием SpaCy и Sentencizer."""
    text = preprocess_text(text)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError(
            "Модель 'en_core_web_sm' не найдена. "
            "Установите её: python -m spacy download en_core_web_sm"
        )

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    random.shuffle(sentences)

    return sentences


def generate_QandA(text: str, num_questions: int = 3, max_tokens: int = 256) -> List[Dict[str, str]]:
    """Генерация вопросов и ответов из текста"""
    if num_questions < 1 or max_tokens < 1:
        raise ValueError("num_questions и max_tokens должны быть >= 1")

    device = 0 if torch.cuda.is_available() else -1

    try:
        model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16
        )

        qg_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer
        )

        qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad",
            device=device
        )

    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели или пайплайнов: {e}")

    sentences = split_text_into_sentences(text)
    results = []

    i = 0
    while i < len(sentences) and len(results) < num_questions:
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue

        # Если предложение слишком короткое и есть следующий, объединяем с ним
        if len(sent.split()) < 10 and i + 1 < len(sentences):
            sent = sent + " " + sentences[i + 1].strip()
            i += 1

        i += 1

        try:
            prompt = f"Read the text and generate one question.\nText: {sent}"
            output = qg_pipeline(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1
            )
            question = output[0]["generated_text"].strip()

            answer_result = qa_pipeline(question=question, context=sent)
            answer = answer_result.get("answer", "").strip()

            results.append({
                "sentence": sent,
                "question": question,
                "answer": answer
            })

        except Exception as e:
            print(f"Ошибка при генерации QA для предложения: {sent}\n{e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Генерация пар вопрос-ответ из текстового файла.")
    parser.add_argument("input_file", help="Путь к входному текстовому файлу.")
    parser.add_argument("--output_file", default=None,
                        help="Путь к выходному файлу. Если не указан, сохраняется в той же папке, что и входной файл.")
    parser.add_argument("--num_questions", type=int, default=3, help="Количество генерируемых пар вопрос-ответ.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Максимальное количество токенов в сгенерированном вопросе.")
    args = parser.parse_args()

    try:
        text = read_text_file(args.input_file)
        results = generate_QandA(text, num_questions=args.num_questions, max_tokens=args.max_tokens)

        if args.output_file is None:
            root, _ = os.path.splitext(args.input_file)
            args.output_file = f"{root}_Q&A.txt"

        write(results, args.output_file)

    except Exception as e:
        print(f"Ошибка при выполнении программы: {e}")


if __name__ == "__main__":
    main()
