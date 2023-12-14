from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialBankInformation,
    FinancialBarcode,
    FinancialCustomerInformation,
    FinancialDocumentInformation,
    FinancialDocumentMetadata,
    FinancialLineItem,
    FinancialLocalInformation,
    FinancialMerchantInformation,
    FinancialParserDataClass,
    FinancialParserObjectDataClass,
    FinancialPaymentInformation
)
def mindee_financial_parser(original_response: dict) -> FinancialParserDataClass:
    extracted_data = []
    for page in original_response["document"]["pages"]:
        customer_information = FinancialCustomerInformation(
            billing_address=page.get("customer_address", {}).get("value")
        )
        merchant_information = FinancialMerchantInformation(

        )
        payment_information = FinancialPaymentInformation(

        )
        document_information = FinancialDocumentInformation(

        )
        bank = FinancialBankInformation(

        )
        local = FinancialLocalInformation(

        )
        metadata = FinancialDocumentMetadata(

        )
        items = []

        extracted_data.append(
            FinancialParserObjectDataClass(
                merchant_information=merchant_information,
                customer_information=customer_information,
                local=local,
                bank=bank,
                document_metadata=metadata,
                financial_document_information=document_information,
                payment_information=payment_information,
                item_lines=items
            )
        )
    return FinancialParserDataClass(extracted_data=extracted_data)