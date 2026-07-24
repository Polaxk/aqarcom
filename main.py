import json
from pathlib import Path

from src.ingestion import ingest_pdf
from src.extractor import extract_features
from src.rules_engine import evaluate_risk
from src.narration import generate_report


def main():
    BASE_DIR = Path(__file__).resolve().parent
    pdf_path = BASE_DIR / "reg2.pdf"

    try:
        # 1. Ingest the PDF.
        raw_text = ingest_pdf(pdf_path)
        print("PDF ingestion successful")
        print(f"Extracted {len(raw_text)} characters\n")

        # 2. Extract structured Aqarcom features.
        features = extract_features(raw_text)

        print("Feature extraction successful\n")
        print("--- Extracted Features ---")
        print(features.model_dump_json(indent=2))

        # 3. Convert Pydantic schema to dictionary.
        feature_data = features.model_dump()

        # 4. Evaluate risk using config/risk_policy.yaml.
        evaluation = evaluate_risk(feature_data)

        print("\n--- Risk Evaluation JSON ---")
        print(json.dumps(evaluation, indent=2))

        # 5. Produce the final report.
        final_report = generate_report(evaluation)

        print("\n--- AQARCOM FINAL RISK REPORT ---")
        print(final_report)

    except FileNotFoundError as err:
        print(f"\nPipeline failed: Missing File -> {err.filename}")

    except Exception as error:
        print(f"\nPipeline failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()