import argparse
import re
import csv
from typing import Dict, List, Tuple


QG_MODEL_ID = "google/flan-t5-large"
QA_MODEL_ID = "distilbert-base-uncased-distilled-squad"


def preprocess_text(text: str) -> str:
    """Очистка текста от лишних пробелов и переносов строк"""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_into_sentences(text: str) -> List[str]:
    """Сегментация текста на предложения с использованием SpaCy и Sentencizer."""
    text = preprocess_text(text)
    if not text:
        return []

    import spacy

    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def load_pipelines() -> Tuple[object, object]:
    """Загружает пайплайны генерации вопросов и поиска ответов."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(QG_MODEL_ID)
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_ID, **model_kwargs)

    qg_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    qa_pipeline = pipeline(
        "question-answering",
        model=QA_MODEL_ID,
        device=device
    )

    return qg_pipeline, qa_pipeline


def generate_QandA(topic: str, num_questions: int = 3, max_tokens: int = 256) -> List[Dict[str, str]]:
    """Генерация вопросов и ответов по теме"""
    if num_questions < 1 or max_tokens < 1:
        raise ValueError("num_questions и max_tokens должны быть >= 1")
    topic = preprocess_text(topic)
    if not topic:
        raise ValueError("Тема не должна быть пустой.")

    try:
        qg_pipeline, qa_pipeline = load_pipelines()
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели или пайплайнов: {e}")

    sentences = split_text_into_sentences(topic)
    if not sentences:
        sentences = [topic]

    results = []
    attempts = 0
    while len(results) < num_questions and attempts < num_questions * 3:
        sent = sentences[attempts % len(sentences)].strip()
        attempts += 1
        if not sent:
            continue

        try:
            prompt = (
                "Generate one clear question about the topic and make it answerable from the context.\n"
                f"Topic/context: {sent}"
            )
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

    if not results:
        raise RuntimeError("Не удалось сгенерировать ни одной пары вопрос-ответ.")

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
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
