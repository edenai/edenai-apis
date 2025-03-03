from typing import Dict, Any, List

from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialMerchantInformation,
    FinancialCustomerInformation,
    FinancialPaymentInformation,
    FinancialDocumentInformation,
    FinancialBankInformation,
    FinancialLocalInformation,
    FinancialDocumentMetadata,
    FinancialLineItem,
    FinancialParserObjectDataClass,
    FinancialParserDataClass,
)
from edenai_apis.utils.conversion import convert_string_to_number


# *****************************financial parser***************************************************
def organize_response_per_pages(
    original_response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Organize dataleon original response into a list of dictionaries, each representing a page

    Args:
        original_response (dict): The original response containing metadata, entities, and tables.

    Returns:
        List[dict]: A list of dictionaries, each representing a page with organized entities and tables.
    """
    organized_pages = []

    # Iterate through each page index
    for page_index in range(
        1, original_response["metadata"]["documents"][0]["pages"] + 1
    ):
        page_dict = {"items": []}

        # Extract entities for the current page
        page_dict.update(
            extract_entities_for_page(original_response["entities"], page_index)
        )

        # Extract tables for the current page
        page_dict["items"] = extract_tables_for_page(
            original_response["tables"], page_index
        )

        organized_pages.append(page_dict)

    return organized_pages


def extract_entities_for_page(
    entities: List[Dict[str, Any]], page_index: int
) -> Dict[str, Any]:
    """
    Extracts entities that belong to a specific page.

    Args:
        entities (list): List of entities from the original response.
        page_index (int): Index of the current page.

    Returns:
        dict: Dictionary containing entities for the specified page.
    """
    page_entities = {}
    for entity in entities:
        if entity["page"] == page_index:
            key_name = entity["name"]
            key_value = entity["value"]
            page_entities[key_name] = key_value
            page_entities["page_index"] = page_index

    return page_entities


def extract_tables_for_page(
    tables: List[Dict[str, Any]], page_index: int
) -> List[Dict[str, Any]]:
    """
    Extracts items and their rows for a specific page.

    Args:
        tables (list): List of tables from the original response.
        page_index (int): Index of the current page.

    Returns:
        List[dict]: List of dictionaries representing table rows for the specified page.
    """
    page_tables = []
    for table in tables:
        if table["page"] == page_index:
            for row in table["rows"]:
                result_dict = {cell["name"]: cell["value"] for cell in row["cells"]}
                page_tables.append(result_dict)

    return page_tables


def dataleon_financial_parser(original_response: Dict) -> FinancialParserDataClass:
    """
    Parses data obtained from the Dataleon financial parser into a structured format.

    Args:
        original_response (dict): Raw data obtained from the Dataleon financial parser.

    Returns:
        FinancialParserDataClass: Structured data object containing financial information.
    """
    extracted_data = []
    formatted_response = organize_response_per_pages(original_response)

    for document_page in formatted_response:
        extracted_data.append(
            FinancialParserObjectDataClass(
                customer_information=FinancialCustomerInformation(
                    name=document_page.get("CustomerName"),
                    billing_address=document_page.get("CustomerAddress"),
                    siren_number=document_page.get("CustomerSIREN"),
                    siret_number=document_page.get("CustomerSIRET"),
                    tax_id=document_page.get("CustomerTVANumber"),
                ),
                merchant_information=FinancialMerchantInformation(
                    name=document_page.get("VendorName"),
                    address=document_page.get("VendorAddress"),
                    siren_number=document_page.get("VendorSIREN"),
                    siret_number=document_page.get("VendorSIRET"),
                    tax_id=document_page.get("VendorTVANumber"),
                ),
                payment_information=FinancialPaymentInformation(
                    subtotal=document_page.get("Subtotal"),
                    total_tax=document_page.get("Tax"),
                    total=document_page.get("Total"),
                    tax_rate=document_page.get("Taux"),
                ),
                financial_document_information=FinancialDocumentInformation(
                    invoice_receipt_id=document_page.get("ID"),
                    invoice_due_date=document_page.get("DueDate"),
                    invoice_date=document_page.get("IssueDate"),
                ),
                bank=FinancialBankInformation(),
                local=FinancialLocalInformation(),
                document_metadata=FinancialDocumentMetadata(
                    document_page_number=document_page.get("page_index")
                ),
                item_lines=[
                    FinancialLineItem(
                        description=item.get("Label"),
                        quantity=convert_string_to_number(item.get("Quantity"), float),
                        sub_total=item.get("SubTotal"),
                        unit_price=item.get("UnitPrice"),
                    )
                    for item in document_page.get("items", [])
                ],
            )
        )

    return FinancialParserDataClass(extracted_data=extracted_data)
