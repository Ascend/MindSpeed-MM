import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.tasks.evaluation.utils.string_utils import process_answer

from .impl_base import BaseEvaluator


class VQA2Evaluator(BaseEvaluator):
    def __init__(self, result_output_path: str, model_name: str, dataset_name: str) -> None:
        self.result_output_path = Path(result_output_path)
        self.result_prefix = f"{model_name}_{dataset_name}"
        self.predictions: list[dict[str, Any]] = []
        self.scores: list[float] = []
        self.answer_type_scores: defaultdict[str, list[float]] = defaultdict(list)
        self.question_type_scores: defaultdict[str, list[float]] = defaultdict(list)

    @staticmethod
    def _score_answer(prediction: str, answers: list[str]) -> float:
        normalized_prediction = process_answer(prediction)
        normalized_answers = [process_answer(answer) for answer in answers]
        accuracies = []
        for index in range(len(normalized_answers)):
            other_answers = normalized_answers[:index] + normalized_answers[index + 1:]
            matching_answers = sum(answer == normalized_prediction for answer in other_answers)
            accuracies.append(min(1.0, matching_answers / 3.0))
        return sum(accuracies) / len(accuracies)

    @staticmethod
    def _average(scores: list[float]) -> float:
        return 100.0 * sum(scores) / len(scores) if scores else 0.0

    def update(self, item: dict[str, Any], prediction: str) -> None:
        answer = prediction.strip()
        score = self._score_answer(answer, item["answers"])
        self.predictions.append({"question_id": item["question_id"], "answer": answer})
        self.scores.append(score)
        self.answer_type_scores[item["answer_type"]].append(score)
        self.question_type_scores[item["question_type"]].append(score)

    def compute(self) -> dict[str, Any]:
        return {
            "overall": round(self._average(self.scores), 2),
            "answer_type": {
                name: round(self._average(scores), 2)
                for name, scores in sorted(self.answer_type_scores.items())
            },
            "question_type": {
                name: round(self._average(scores), 2)
                for name, scores in sorted(self.question_type_scores.items())
            },
        }

    def finalize(self) -> dict[str, Any]:
        metrics = self.compute()
        self.result_output_path.mkdir(parents=True, exist_ok=True)
        predictions_path = self.result_output_path / f"{self.result_prefix}_predictions.json"
        metrics_path = self.result_output_path / f"{self.result_prefix}_metrics.json"
        with predictions_path.open("w", encoding="utf-8") as file:
            json.dump(self.predictions, file, ensure_ascii=False, indent=2)
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        print_rank(print, "\n===== Evaluation Summary =====")
        print_rank(print, f"Predictions saved to: {predictions_path}")
        print_rank(print, f"Metrics saved to: {metrics_path}")
        return metrics
