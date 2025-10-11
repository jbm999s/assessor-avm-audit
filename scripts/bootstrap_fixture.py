
from pathlib import Path
import shutil

src = Path("fixtures/sample_model_assessment_pin.parquet")
dst = Path("output/assessment_pin/model_assessment_pin.parquet")

dst.parent.mkdir(parents=True, exist_ok=True)
if dst.exists():
    print(f"Exists: {dst} (no action)")
elif src.exists():
    shutil.copy2(src, dst)
    print(f"Bootstrapped {dst} from {src}")
else:
    print("No fixture found at fixtures/sample_model_assessment_pin.parquet; skipping.")
