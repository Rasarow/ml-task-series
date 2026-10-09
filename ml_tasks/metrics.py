import re


def normalize_words(text):
    return " ".join(re.findall("\\w+", text.lower(), flags=re.UNICODE))


def rouge(reference, prediction):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return {
        name + "_f1": round(score.fmeasure, 6)
        for name, score in scorer.score(reference, prediction).items()
    }


def speech_metrics(references, predictions):
    from jiwer import process_words, cer

    if not references or len(references) != len(predictions):
        raise ValueError("Provide equally sized nonempty reference and prediction lists.")
    references = [normalize_words(text) for text in references]
    predictions = [normalize_words(text) for text in predictions]
    words = process_words(references, predictions)
    return {
        "normalized_wer": round(words.wer, 6),
        "normalized_cer": round(cer(references, predictions), 6),
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "reference_words": words.hits + words.substitutions + words.deletions,
    }
