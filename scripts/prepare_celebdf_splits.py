from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.celebdf_prepare import prepare_celebdf_splits


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "celebdf_prepare_config.json"
    summary = prepare_celebdf_splits(config_path)

    print("\nCeleb-DF balanced selection and split completed successfully.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()