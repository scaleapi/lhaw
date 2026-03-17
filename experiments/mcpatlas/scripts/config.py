"""Configuration system for underspecification dataset preparation."""

from pathlib import Path
from typing import Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class BaselineConfig(BaseModel):
    """Configuration for baseline discovery and aggregation."""

    data_dir: str = Field(
        default="full_inference_data", description="Directory containing baseline inference results"
    )
    model_patterns: Dict[str, str] = Field(description="Glob patterns to auto-discover model runs")
    pass_threshold: float = Field(
        default=0.75, description="Threshold for considering a task as passed"
    )
    aggregation_strategy: Literal["union", "intersection"] = Field(
        default="intersection", description="Strategy for combining passed tasks across models"
    )


class UnderspecConfig(BaseModel):
    """Configuration for underspecified variant generation."""

    severity: Literal["DELETE", "VAGUIFY", "GENERICIZE"] = Field(
        default="DELETE", description="Removal strategy for underspecification"
    )
    top_k: Optional[int] = Field(
        default=None, description="Limit to top K segments by priority (None = all)"
    )
    variant_top_k: int = Field(default=1, description="Number of variants to include per task")
    min_level: int = Field(
        default=1,
        description="Min segments to remove (filters out variants with fewer segments removed)",
    )
    max_level: int = Field(
        default=1, description="Max segments to remove together (1=single, 2=pairs, etc.)"
    )
    extraction_model: str = Field(
        default="gemini/gemini-3-flash-preview", description="Model for segment extraction"
    )
    generation_model: str = Field(
        default="gemini/gemini-3-flash-preview", description="Model for variant generation"
    )


class DatasetConfig(BaseModel):
    """Top-level configuration for dataset preparation pipeline."""

    dataset: Dict[str, str] = Field(description="Dataset metadata (name, description)")
    baseline: BaselineConfig = Field(description="Baseline discovery and aggregation configuration")
    underspec: UnderspecConfig = Field(
        description="Underspecified variant generation configuration"
    )
    output: Dict[str, str] = Field(description="Output configuration (base_dir, etc.)")

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
