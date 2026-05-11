import sys
import os
from pathlib import Path
from typing import ClassVar
from pydantic import BaseModel, Field

if 'google.colab' in sys.modules:
    _REPO_ROOT = Path("/content/repo")
    _DATA_PATH = _REPO_ROOT / "IB_DL" / "data"
    _OUTPUT_PATH = Path("/content/drive/MyDrive/IB_DL_RESULTS")
    _OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

else:
    _REPO_ROOT = Path.cwd()
    _DATA_PATH = _REPO_ROOT / "IB_DL" / "data"
    _OUTPUT_PATH = _REPO_ROOT


class PathsConfig(BaseModel):
    data_dir: ClassVar[Path] = _DATA_PATH
    output_dir: ClassVar[Path] = _OUTPUT_PATH


class DefaultLabels(BaseModel):
    class_labels: list[str] = Field(
        default_factory=lambda: ["healthy", "COPD", "ILD"]
    )
    num_labels: list[int] = Field(
        default_factory=lambda: [0, 1, 2]
    )
