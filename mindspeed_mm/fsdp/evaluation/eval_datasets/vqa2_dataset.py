import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


class VQA:
    """Load and index VQA question and annotation files.
    """

    def __init__(
        self,
        annotation_file: Optional[str | Path] = None,
        question_file: Optional[str | Path] = None,
    ) -> None:
        self.dataset: dict[str, Any] = {}
        self.questions: dict[str, Any] = {}
        self.qa: dict[int, dict[str, Any]] = {}
        self.qqa: dict[int, dict[str, Any]] = {}
        self.imgToQA: dict[int, list[dict[str, Any]]] = {}

        if annotation_file is None or not Path(annotation_file).is_file():
            raise FileNotFoundError(f"VQA annotation file does not exist: {annotation_file}")
        if question_file is None or not Path(question_file).is_file():
            raise FileNotFoundError(f"VQA question file does not exist: {question_file}")

        self.dataset = self._load_json(annotation_file)
        self.questions = self._load_json(question_file)
        self.createIndex()

    @staticmethod
    def _load_json(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def createIndex(self) -> None:
        """Create the indexes exposed by the official VQA helper."""
        img_to_qa: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        qa: dict[int, dict[str, Any]] = {}
        qqa: dict[int, dict[str, Any]] = {}

        for annotation in self.dataset.get("annotations", []):
            question_id = annotation["question_id"]
            image_id = annotation["image_id"]
            qa[question_id] = annotation
            img_to_qa[image_id].append(annotation)

        for question in self.questions.get("questions", []):
            question_id = question["question_id"]
            qqa[question_id] = question

        self.imgToQA = dict(img_to_qa)
        self.qa = qa
        self.qqa = qqa

    @staticmethod
    def _as_list(values: Optional[int | str | Iterable[int | str]]) -> list[int | str]:
        if values is None:
            return []
        if isinstance(values, (int, str)):
            return [values]
        return list(values)

    def getQuesIds(
        self,
        imgIds: Optional[int | Iterable[int]] = None,
        quesTypes: Optional[str | Iterable[str]] = None,
        ansTypes: Optional[str | Iterable[str]] = None,
    ) -> list[int]:
        img_ids = self._as_list(imgIds)
        question_types = self._as_list(quesTypes)
        answer_types = self._as_list(ansTypes)
        annotations = self.dataset.get("annotations", [])

        if img_ids:
            annotations = [annotation for image_id in img_ids for annotation in self.imgToQA.get(image_id, [])]
        if question_types:
            annotations = [item for item in annotations if item.get("question_type") in question_types]
        if answer_types:
            annotations = [item for item in annotations if item.get("answer_type") in answer_types]
        return [item["question_id"] for item in annotations]


class VQA2ValDataset:
    QUESTIONS_FILE = "v2_OpenEnded_mscoco_val2014_questions.json"
    ANNOTATIONS_FILE = "v2_mscoco_val2014_annotations.json"
    PROMPT_SUFFIX = "Answer the question using a single word or phrase."

    def __init__(self, dataset_root: str | Path, max_samples: Optional[int] = None) -> None:
        root = Path(dataset_root)
        self.questions_path = root / self.QUESTIONS_FILE
        self.annotations_path = root / self.ANNOTATIONS_FILE
        self.images_dir = root / "val2014"
        self.vqa = VQA(
            annotation_file=str(self.annotations_path),
            question_file=str(self.questions_path),
        )
        self.question_ids = self.vqa.getQuesIds()
        if max_samples is not None:
            self.question_ids = self.question_ids[:max_samples]

    def __len__(self) -> int:
        return len(self.question_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        question_id = self.question_ids[index]
        question = self.vqa.qqa[question_id]
        annotation = self.vqa.qa[question_id]
        image_id = question["image_id"]
        question_text = question["question"].strip()
        return {
            "question_id": question_id,
            "image_id": image_id,
            "image": str(self.images_dir / f"COCO_val2014_{image_id:012d}.jpg"),
            "text": f"{question_text}\n{self.PROMPT_SUFFIX}",
            "question": question_text,
            "answers": [answer["answer"] for answer in annotation["answers"]],
            "multiple_choice_answer": annotation["multiple_choice_answer"],
            "answer_type": annotation["answer_type"],
            "question_type": annotation["question_type"],
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]
