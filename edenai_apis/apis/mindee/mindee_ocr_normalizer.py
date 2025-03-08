from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialBankInformation,
    FinancialCustomerInformation,
    FinancialDocumentInformation,
    FinancialDocumentMetadata,
    FinancialLineItem,
    FinancialLocalInformation,
    FinancialMerchantInformation,
    FinancialParserDataClass,
    FinancialParserObjectDataClass,
    FinancialPaymentInformation,
)


def mindee_financial_parser(original_response: dict) -> FinancialParserDataClass:
    """
    Parses data obtained from the Mindee financial parser into a structured format.

    Args:
        original_response (dict): Raw data obtained from the Mindee financial parser.

    Returns:
        FinancialParserDataClass: Structured data object containing financial information.
    """
    extracted_data = []
    for idx, page in enumerate(original_response["document"]["inference"]["pages"]):
        predictions = page.get("prediction", {})
        customer_company_registrations = predictions.get(
            "customer_company_registrations", []
        )
        customer_infos = {}
        for customer_info in customer_company_registrations:
            customer_type = customer_info.get("type", "")
            customer_registration_value = customer_info.get("value", "")
            customer_infos.update({customer_type: customer_registration_value})

        merchant_company_registrations = predictions.get(
            "supplier_company_registrations", []
        )
        supplier_infos = {}
        for supplier_info in merchant_company_registrations:
            supplier_type = supplier_info.get("type", "")
            supplier_registration_value = supplier_info.get("value", "")
            supplier_infos.update({supplier_type: supplier_registration_value})

        customer_information = FinancialCustomerInformation(
            name=predictions.get("customer_name", {}).get("value"),
            billing_address=predictions.get("customer_address", {}).get("value"),
            vat_number=customer_infos.get("VAT NUMBER"),
            abn_number=customer_infos.get("ABN"),
            gst_number=customer_infos.get("GST/HST"),
            siren_number=customer_infos.get("SIREN"),
            siret_number=customer_infos.get("SIRET"),
            tax_id=customer_infos.get("TAX ID"),
            registration_number=customer_infos.get("COMPANY REGISTRATION NUMBER"),
        )
        merchant_information = FinancialMerchantInformation(
            name=predictions.get("supplier_name", {}).get("value"),
            phone=predictions.get("supplier_phone_number", {}).get("value"),
            address=predictions.get("supplier_address", {}).get("value"),
            vat_number=supplier_infos.get("VAT NUMBER"),
            abn_number=supplier_infos.get("ABN"),
            gst_number=supplier_infos.get("GST/HST"),
            siren_number=supplier_infos.get("SIREN"),
            siret_number=supplier_infos.get("SIRET"),
            tax_id=supplier_infos.get("TAX ID"),
            registration_number=supplier_infos.get("COMPANY REGISTRATION NUMBER"),
        )
        payment_information = FinancialPaymentInformation(
            total_tax=predictions.get("total_tax", {}).get("value"),
            amount_tip=predictions.get("tip", {}).get("value"),
            total=predictions.get("total_amount", {}).get("value"),
            subtotal=predictions.get("total_net", {}).get("value"),
        )
        document_information = FinancialDocumentInformation(
            invoice_date=predictions.get("date", {}).get("value"),
            invoice_due_date=predictions.get("due_date", {}).get("value"),
            invoice_receipt_id=predictions.get("invoice_number", {}).get("value"),
            time=predictions.get("time", {}).get("value"),
        )
        bank = FinancialBankInformation()

        local = FinancialLocalInformation(
            currency=predictions.get("local", {}).get("currency"),
            language=predictions.get("local", {}).get("language"),
        )
        metadata = FinancialDocumentMetadata(
            document_type=predictions.get("document_type", {}).get("value"),
            document_page_number=idx + 1,
        )
        items = [
            FinancialLineItem(
                description=item.get("description"),
                product_code=item.get("product_code"),
                quantity=item.get("quantity"),
                amount_line=item.get("total_amount"),
                tax=item.get("tax_amount"),
                tax_rate=item.get("tax_rate"),
                unit_price=item.get("unit_price"),
            )
            for item in predictions.get("line_items")
        ]

        extracted_data.append(
            FinancialParserObjectDataClass(
                merchant_information=merchant_information,
                customer_information=customer_information,
                local=local,
                bank=bank,
                document_metadata=metadata,
                financial_document_information=document_information,
                payment_information=payment_information,
                item_lines=items,
            )
        )
    return FinancialParserDataClass(extracted_data=extracted_data)
