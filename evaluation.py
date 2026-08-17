from pandas import DataFrame
from torch import nn
from PIL import Image
import os
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from generation import generate_caption_beam_search
from vocabulary import Vocabulary


def _clean_caption_for_metric(caption: str) -> str:
    return caption.replace("|||", " ").replace("\n", " ").replace("\r", " ").strip()


def compute_caption_metrics(
    ground_truths: dict[str, list[str]],
    predictions: dict[str, str],
) -> dict[str, float]:
    clean_ground_truths = {
        key: [_clean_caption_for_metric(caption) for caption in captions]
        for key, captions in ground_truths.items()
    }
    predictions_for_metric = {
        key: [_clean_caption_for_metric(caption)]
        for key, caption in predictions.items()
    }

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr")
    ]

    metric_scores = {}
    for scorer, metric in scorers:
        score, _scores = scorer.compute_score(clean_ground_truths, predictions_for_metric)
        if isinstance(metric, list):
            for metric_name, metric_score in zip(metric, score):
                metric_scores[metric_name] = metric_score
        else:
            metric_scores[metric] = score
            
    return metric_scores
    

def evaluate_caption_metrics(
        encoder: nn.Module,
        decoder: nn.Module,
        vocab: Vocabulary,
        df: DataFrame,
        images_dir: str,
        image_transform,
        max_length: int,
        beam_size: int,
        length_penalty: float,
        limit: int | None = None,
        show_progress: bool = True
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, float]]:
    encoder.eval()
    decoder.eval()
    
    references: dict[str, list[str]] = {}
    predictions: dict[str, str] = {}

    eval_df = df.head(limit) if limit is not None else df
    pbar = tqdm(eval_df.iterrows(), desc="Evaluating", total=len(eval_df), leave=False, disable=not show_progress)

    for _idx, row in pbar:
        filename = row["filename"]
        filepath = row["filepath"]
        sentences = row["sentences"]

        image_path = os.path.join(images_dir, filepath, filename)
        with Image.open(image_path) as img:
            image = image_transform(img.convert("RGB"))

        caption = generate_caption_beam_search(encoder, decoder, image, vocab, max_length, beam_size, length_penalty)

        predictions[filename] = caption
        references[filename] = [sentence["raw"] for sentence in sentences]

    metric_scores = compute_caption_metrics(references, predictions)

    return predictions, references, metric_scores
