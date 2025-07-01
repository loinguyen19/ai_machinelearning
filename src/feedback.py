from pathlib import Path
from typing import Annotated, List, Optional
from pydantic import BaseModel, AfterValidator, Field
import json
import os


def validate_item_label(value: str) -> str:
    valid_labels = ["dish", "tray"]
    if value not in valid_labels:
        raise ValueError(f"Label must be one of {valid_labels}")
    return value


def validate_classification(value: str) -> str:
    valid_classifications = ["empty", "kakigori", "not_empty"]
    if value not in valid_classifications:
        raise ValueError(f"Classification must be one of {valid_classifications}")
    return value


class Feedback(BaseModel):
    frame_id: int = Field(ge=0)
    item_id: int = Field(ge=0)
    user_label: Annotated[str, AfterValidator(validate_item_label)]
    classification: Annotated[str, AfterValidator(validate_classification)]
    correct: bool
    comments: Optional[str] = None

    model_config = {"strict": True}


class FeedbackLog:
    def __init__(self, log_file: str = None):
        project_root = Path(__file__).parent.parent if os.path.basename(os.getcwd()) == 'src' else Path(os.getcwd())
        self.log_file = os.getenv("FEEDBACK_LOG_PATH", log_file or str(project_root / "feedback_log.json"))
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.feedback_list: List[Feedback] = []
        if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_list = [Feedback(**item) for item in data]
            except json.JSONDecodeError:
                print(f"Warning: {self.log_file} contains invalid JSON. Initializing empty feedback list.")
                self.feedback_list = []
                # Initialize with empty JSON array
                with open(self.log_file, 'w') as f:
                    json.dump([], f, indent=2)
        else:
            # Create empty JSON file if it doesn't exist or is empty
            with open(self.log_file, 'w') as f:
                json.dump([], f, indent=2)

    def add_feedback(self, feedback: Feedback):
        self.feedback_list.append(feedback)
        with open(self.log_file, 'w') as f:
            json.dump([item.dict() for item in self.feedback_list], f, indent=2)

    def get_misclassified_samples(self) -> List[Feedback]:
        return [fb for fb in self.feedback_list if not fb.correct]

