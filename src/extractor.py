import json
import os

from openai import OpenAI
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into environment variables

API_KEY = os.getenv("NVIDIA_API_KEY")

class ExtractedFeatures(BaseModel):
    case_id: str
    customer_id: str
    customer_type: str
    account_type: str
    account_status: str
    debit_locked_flag: bool
    credit_locked_flag: bool
    current_balance_jod: float
    balance_band: str

    property_type: str
    parcel_area_m2: float
    zoning_category: str
    intended_use: str
    zoning_mismatch_flag: bool
    transaction_value_jod: float

    title_seller_match_flag: bool
    missing_required_docs_count: int
    foreign_buyer_flag: bool
    foreign_approval_docs_present_flag: bool
    document_completeness_score: float = Field(ge=0, le=100)


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)


def extract_features(raw_text: str) -> ExtractedFeatures:
    schema = ExtractedFeatures.model_json_schema()

    prompt = f"""
You are Aqarcom's deterministic financing-case extraction engine.

TASK:
Extract information from SOURCE CASE TEXT and return exactly one JSON object
that conforms to the schema below.

NON-NEGOTIABLE OUTPUT RULES:
- Return valid JSON only. No markdown, no code fences, no explanations, no reasoning.
- Include every schema field exactly once.
- Do not add fields.
- Use only facts explicitly stated in SOURCE CASE TEXT.
- Never infer, guess, or invent a name, ID, amount, document, account state, or legal fact.

MISSING-VALUE RULES:
- Missing string: "unknown"
- Missing numeric value: 0
- Missing boolean: false
- document_completeness_score: integer from 0 to 100

NORMALIZATION RULES:
- customer_type: individual | corporate | business | unknown
- account_type: savings | current | salary | checking | payroll | merchant | operating | unknown
- account_status: active | suspended | closed | unknown
- balance_band: zero | negative | low | normal | high | unknown
- property_type: land | commercial | industrial | warehouse | office | retail | residential | mixed_use | unknown
- zoning_category: residential | commercial | industrial | light_industrial | agricultural | mixed_use | unknown
- Money fields must be numeric JOD values only: remove commas, currency symbols, and words.
- Convert clearly stated property area to square metres where possible.

BOOLEAN RULES:
- zoning_mismatch_flag is true only when the text explicitly establishes that intended use conflicts with zoning.
- title_seller_match_flag is true only when the transaction seller explicitly matches the stated title-holder/owner.
- foreign_buyer_flag is true only when foreign ownership/buyer status is explicit.
- foreign_approval_docs_present_flag is true only when a foreign-ownership approval document is explicitly present.
- If foreign_buyer_flag is false, foreign_approval_docs_present_flag must be false.
- Count only explicitly missing required documents.

SAFETY:
- Do not make a loan decision.
- Do not claim legal compliance, title verification, or ownership verification.
- This is extraction only.

JSON SCHEMA:
{json.dumps(schema, indent=2)}

SOURCE CASE TEXT:
---START SOURCE---
{raw_text}
---END SOURCE---
"""

    try:
        response = client.chat.completions.create(
            model="z-ai/glm-5.2",
            messages=[
                {
                    "role": "system",
                    "content": "Return one valid JSON object only. Never include prose or markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            top_p=1,
            max_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extracted_features",
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        content = response.choices[0].message.content
        return ExtractedFeatures.model_validate_json(content)

    except KeyError as error:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. Set it before running main.py."
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"Feature extraction failed through NVIDIA NIM: {error}"
        ) from error