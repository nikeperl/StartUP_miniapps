import argparse
import random
import re
import csv
from typing import List, Dict

import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


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


def generate_QandA(topic: str, num_questions: int = 3, max_tokens: int = 256) -> List[Dict[str, str]]:
    """Генерация вопросов и ответов по теме"""
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

    # Разбиваем тему на "предложения" (можно просто использовать саму тему)
    sentences = split_text_into_sentences(topic)
    if not sentences:
        sentences = [topic]

    results = []
    i = 0
    while i < len(sentences) and len(results) < num_questions:
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue

        # Объединяем короткие предложения
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


def write_csv(results: List[Dict[str, str]], output_file: str = "output.csv") -> None:
    """Сохраняет результаты в CSV"""
    try:
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["sentence", "question", "answer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in results:
                writer.writerow(item)
        print(f"Результаты записаны в {output_file}")
    except Exception as e:
        print(f"Ошибка при записи CSV: {e}")


def main():
    parser = argparse.ArgumentParser(description="Генерация пар вопрос-ответ по теме.")
    parser.add_argument("topic", help="Тема для генерации вопросов и ответов.")
    parser.add_argument("--output_file", default="output.csv",
                        help="Путь к выходному CSV файлу.")
    parser.add_argument("--num_questions", type=int, default=3, help="Количество генерируемых пар вопрос-ответ.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Максимальное количество токенов в сгенерированном вопросе.")
    args = parser.parse_args()

    try:
        results = generate_QandA(args.topic, num_questions=args.num_questions, max_tokens=args.max_tokens)
        write_csv(results, args.output_file)
    except Exception as e:
        print(f"Ошибка при выполнении программы: {e}")


if __name__ == "__main__":
    main()
