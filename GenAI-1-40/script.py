import argparse
import os
import sys


# Загрузка модели
model = None
def get_model():
    global model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[ОШИБКА] Не удалось загрузить модель SentenceTransformer: {e}", file=sys.stderr)
            sys.exit(1)
    return model

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
                from PyPDF2 import PdfReader

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
            import docx2txt

            text = docx2txt.process(filepath)
            if text is None:
                text = ""
            return text

        else:
            raise ValueError(f"Формат {ext} не поддерживается. Используй txt, pdf, docx.")

    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении файла '{filepath}': {e}")

def chunk_text(text: str, max_tokens: int = 500) -> list:
    """Разбивает текст на части (для обработки больших файлов)."""
    if max_tokens < 1:
        raise ValueError("max_tokens должен быть >= 1")

    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

def encode_and_save(filepath: str):
    """ Кодирует документ и сохраняет результат в файл. """
    try:
        text = read_document(filepath)
    except Exception as e:
        print(f"[ОШИБКА] {e}", file=sys.stderr)
        return

    if not text.strip():
        print(f"[ОШИБКА] Файл '{filepath}' пустой или текст не удалось извлечь.", file=sys.stderr)
        return

    root, _ = os.path.splitext(filepath)
    out = f"{root}_emb.npy"

    try:
        import numpy as np
        import torch

        model = get_model()
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Нет текста для построения эмбеддинга")

        embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                emb = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
                embeddings.append(emb) 

        # Средний эмбединг
        embedding = np.mean(embeddings, axis=0)

        if np.isnan(embedding).any():
            raise ValueError("Эмбеддинг содержит NaN")

        np.save(out, embedding)
        print(f"Эмбеддинг сохранён в {out}")
    except Exception as e:
        print(f"[ОШИБКА] Не удалось создать эмбеддинг: {e}", file=sys.stderr)


def cosine_similarity(file1: str, file2: str) -> float:
    """Считает косинусное сходство между эмбеддингами из двух файлов."""
    import numpy as np

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
    if emb1.shape != emb2.shape:
        raise ValueError(f"Размеры эмбеддингов не совпадают: {emb1.shape} и {emb2.shape}.")

    try:
        denominator = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if denominator == 0:
            raise ValueError("Невозможно вычислить сходство для нулевого вектора.")
        similarity = np.dot(emb1, emb2) / denominator
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
