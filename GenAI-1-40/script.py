from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import argparse
import os
import sys

# Для чтения разных форматов
import docx2txt
from PyPDF2 import PdfReader


# Загрузка модели
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"[ОШИБКА] Не удалось загрузить модель SentenceTransformer: {e}", file=sys.stderr)
    sys.exit(1)


def read_document(filepath: str) -> str:
    """Считывание текста из txt, pdf, docx."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Файл '{filepath}' не найден.")

    root, ext = os.path.splitext(filepath)
    ext = ext.lower()

    try:
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".pdf":
            text = []
            try:
                with open(filepath, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text.append(extracted)
            except Exception as e:
                raise ValueError(f"Ошибка при чтении PDF '{filepath}': {e}")
            return "\n".join(text)

        elif ext == ".docx":
            text = docx2txt.process(filepath)
            if text is None:
                text = ""
            return text

        else:
            raise ValueError(f"Формат {ext} не поддерживается. Используй txt, pdf, docx.")

    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении файла '{filepath}': {e}")


def encode_and_save(filepath: str):
    """ Кодирует документ и сохраняет результат в файл. """
    try:
        text = read_document(filepath)
    except Exception as e:
        print(f"[ОШИБКА] {e}", file=sys.stderr)
        return

    if not text.strip():
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Файл '{filepath}' пустой. Эмбеддинг может быть бесполезным.", file=sys.stderr)

    root, _ = os.path.splitext(filepath)
    out = f"{root}_emb.npy"

    try:
        with torch.no_grad():
            embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        np.save(out, embedding)
        print(f"Эмбеддинг сохранён в {out}")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось создать эмбеддинг: {e}", file=sys.stderr)


def cosine_similarity(file1: str, file2: str) -> float:
    """Считает косинусное сходство между эмбеддингами из двух файлов."""
    for f in (file1, file2):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Файл '{f}' не найден.")

    try:
        emb1 = np.load(file1)
        emb2 = np.load(file2)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки эмбеддингов: {e}")

    if emb1.size == 0 or emb2.size == 0:
        raise ValueError("Один из эмбеддингов пустой.")

    try:
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception as e:
        raise RuntimeError(f"Ошибка при вычислении косинусного сходства: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Эмбеддинги документов и косинусное сходство")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Подкоманда encode
    encode_parser = subparsers.add_parser("encode", help="Закодировать документ и сохранить эмбеддинг")
    encode_parser.add_argument("--file", type=str, required=True, help="Документ (txt, pdf, docx)")

    # Подкоманда similarity
    sim_parser = subparsers.add_parser("similarity", help="Вычислить косинусное сходство")
    sim_parser.add_argument("--file1", type=str, required=True, help="Файл первого эмбеддинга")
    sim_parser.add_argument("--file2", type=str, required=True, help="Файл второго эмбеддинга")

    args = parser.parse_args()

    if args.command == "encode":
        encode_and_save(args.file)

    elif args.command == "similarity":
        try:
            score = cosine_similarity(args.file1, args.file2)
            print(f"Косинусное сходство: {score:.4f}")
        except Exception as e:
            print(f"[ОШИБКА] {e}", file=sys.stderr)
