from src.ingestion import ingest_pdf
from src.extractor import extract_features
from src.rules_engine import evaluate_risk
from src.narration import generate_report

from pathlib import Path
def main():
    

    # Dynamically gets the exact directory where main.py is located
    BASE_DIR = Path(__file__).resolve().parent

    # Automatically points to the PDF sitting right next to main.py
    pdf_path = BASE_DIR / "Attachment N1 - Sample Loan Agreement.pdf"
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

        # 3. Convert the Pydantic feature schema into a standard dictionary
        #    for the deterministic policy-based risk engine.
        feature_data = features.model_dump()

        # 4. Evaluate risk using config/risk_policy.yaml.
        evaluation = evaluate_risk(feature_data)

        print("\n--- Risk Evaluation JSON ---")
        import json
        print(json.dumps(evaluation, indent=2))

        # 5. Produce the deterministic, presentation-ready report.
        final_report = generate_report(evaluation)

        print("\n--- AQARCOM FINAL RISK REPORT ---")
        print(final_report)

    except FileNotFoundError:
        print(f"Pipeline failed: PDF file not found:\n{pdf_path}")

    except Exception as error:
        print(f"Pipeline failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()