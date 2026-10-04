from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.celebdf_frame_extract import extract_celebdf_images


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "celebdf_extract_frames_config.json"
    summary = extract_celebdf_images(config_path)

    print("\nCeleb-DF image extraction completed successfully.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()