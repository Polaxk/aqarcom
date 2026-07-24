from docling.document_converter import DocumentConverter


def ingest_pdf(file_path: str) -> str:
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)

        text = result.document.export_to_markdown().strip()

        if not text:
            raise ValueError("No readable text found in the PDF.")

        return text

    except Exception as error:
        raise ValueError(
            f"Could not ingest PDF. Use a valid digital/text-based PDF. Error: {error}"
        )


