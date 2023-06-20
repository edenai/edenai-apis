from io import BufferedReader
from json import JSONDecodeError
from typing import Dict, List

import requests
from edenai_apis.features import ProviderInterface, OcrInterface
from edenai_apis.features.ocr.invoice_parser import InvoiceParserDataClass
from edenai_apis.features.ocr.invoice_parser.invoice_parser_dataclass import BankInvoice, CustomerInformationInvoice, InfosInvoiceParserDataClass, ItemLinesInvoice, LocaleInvoice, MerchantInformationInvoice, TaxesInvoice
from edenai_apis.features.ocr.receipt_parser import ReceiptParserDataClass
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import CustomerInformation, InfosReceiptParserDataClass, ItemLines, Locale, MerchantInformation, PaymentInformation, Taxes
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException
#from edenai_apis.features.ocr.resume_parser import ResumeParserDataClass, ResumeLocation, ResumeSkill, ResumeLang, ResumeWorkExpEntry, ResumeWorkExp, ResumeEducationEntry, ResumeEducation, ResumePersonalName, ResumePersonalInfo, ResumeExtractedData
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    Country,
    InfoCountry,
    ItemIdentityParserDataClass,
    format_date,
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import IdentityParserDataClass
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import InfosIdentityParserDataClass
class KlippaApi(ProviderInterface, OcrInterface):
    provider_name = "klippa"

    def __init__(self, api_keys: Dict = {}):
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name, api_keys = api_keys)
        self.api_key = self.api_settings["subscription_key"]
        self.url = "https://custom-ocr.klippa.com/api/v1/parseDocument"

        self.headers = {
            "X-Auth-Key": self.api_key,
        }

    def _make_post_request(self, file: BufferedReader):
        files = {
            "document": file,
            "pdf_text_extraction": "full",
        }

        response = requests.post(
            url=self.url,
            headers=self.headers,
            files=files
        )

        try:
            original_response = response.json()
        except JSONDecodeError:
            raise ProviderException(message="Internal Server Error", code=500) from JSONDecodeError

        if response.status_code != 200:
            raise ProviderException(message=response.json(), code=response.status_code)

        return original_response

    def ocr__invoice_parser(
        self,
        file: str,
        language: str,
        file_url: str = ""
    ) -> ResponseType[InvoiceParserDataClass]:

        file_ = open(file, "rb")
        original_response = self._make_post_request(file_)

        file_.close()

        data_response = original_response["data"]
        customer_information = CustomerInformationInvoice(
            customer_name=data_response["customer_name"],
            customer_address=data_response["customer_address"],
            customer_email=data_response["customer_email"],
            customer_phone=data_response["customer_phone"],
            customer_tax_id=data_response["customer_vat_number"],
            customer_id=data_response["customer_id"],
            customer_billing_address=data_response["customer_address"],
        )

        merchant_information = MerchantInformationInvoice(
            merchant_name=data_response["merchant_name"],
            merchant_address=data_response["merchant_address"],
            merchant_email=data_response["merchant_email"],
            merchant_phone=data_response["merchant_phone"],
            merchant_tax_id=data_response["merchant_vat_number"],
            merchant_id=data_response["merchant_id"],
            merchant_siret=data_response["merchant_coc_number"],
            merchant_website=data_response["merchant_website"],
        )

        bank_information = BankInvoice(
            account_number=data_response["merchant_bank_account_number"],
            sort_code=data_response["merchant_bank_domestic_bank_code"],
            iban=data_response["merchant_bank_account_number_bic"],
        )

        tax_information = TaxesInvoice(
            tax_rate=data_response["personal_income_tax_rate"],
            tax_amount=data_response["personal_income_tax_amount"],
        )

        locale_information = LocaleInvoice(
            currency=data_response["currency"],
            language=data_response["document_language"],
        )

        item_lines: List[ItemLinesInvoice] = []
        for line in data_response.get('lines', []):
            for item in line.get('lineitems', []):
                item_lines.append(ItemLinesInvoice(
                        description=item["description"],
                        quantity=item["quantity"],
                        unit_price=item["amount_each"],
                        discount=item["discount_amount"],
                        amount=item["amount"],
                        tax_rate=item["vat_percentage"],
                        tax_amount=item["vat_amount"],
                        product_code=item["sku"],
                    )
                )

        standardize_response =InvoiceParserDataClass(extracted_data=[InfosInvoiceParserDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            bank_informations=bank_information,
            taxes=[tax_information],
            locale=locale_information,
            item_lines=item_lines,
            invoice_number=data_response["invoice_number"],
            invoice_date=data_response["date"],
            invoice_total=data_response["amount"],
        )])

        return ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response
        )

    def ocr__receipt_parser(
        self,
        file: str,
        language: str,
        file_url: str= ""
    ) -> ResponseType[ReceiptParserDataClass]:

        file_ = open(file, "rb")
        original_response = self._make_post_request(file_)

        file_.close()

        data_response = original_response["data"]
        customer_information = CustomerInformation(
            customer_name=data_response["customer_name"],
        )

        merchant_information = MerchantInformation(
            merchant_name=data_response["merchant_name"],
            merchant_address=data_response["merchant_address"],
            merchant_phone=data_response["merchant_phone"],
            merchant_tax_id=data_response["merchant_vat_number"],
            merchant_siret=data_response["merchant_coc_number"],
            merchant_url=data_response["merchant_website"],
        )

        locale_information = Locale(
            currency=data_response["currency"],
            language=data_response["document_language"],
            country=data_response["merchant_country_code"],
        )

        taxes_information = Taxes(
            rate=data_response["personal_income_tax_rate"],
            taxes=data_response["personal_income_tax_amount"],
        )

        payment_information = PaymentInformation(
            card_type=data_response["paymentmethod"],
            card_number=data_response["payment_card_number"],
        )

        item_lines: List[ItemLines] = []
        for line in data_response.get('lines', []):
            for lineitem in line.get('linetimes', []):
                item_lines.append(ItemLines(
                        description=lineitem["description"],
                        quantity=lineitem["quantity"],
                        unit_price=lineitem["amount_each"],
                        amount=lineitem["amount"],
                    )
                )

        info_receipt = [InfosReceiptParserDataClass(
            customer_information=customer_information,
            merchant_information=merchant_information,
            locale=locale_information,
            taxes=[taxes_information],
            payment_information=payment_information,
            invoice_number=data_response["invoice_number"],
            date=data_response["date"],
            invoice_total=data_response["amount"],
        )]

        standardize_response = ReceiptParserDataClass(extracted_data=info_receipt)

        return ResponseType[ReceiptParserDataClass](
            original_response=original_response,
            standardized_response=standardize_response
        )
    """
    def ocr__resume_parser(
        self,
        file:str,
        file_url: str = ""
    )-> ResponseType[ResumeParserDataClass]:
        
        file_ = open(file, "rb")
        
        original_response = self._make_post_request(file_)
        file_.close()
        
        data_response = original_response["data"]

        print(data_response)

        personal_info = ResumePersonalInfo(
            name=ResumePersonalName(
                first_name=data_response["first_name"],
                last_name=data_response["last_name"],
                raw_name=data_response["raw_name"],
                middle=data_response["middle"],
                title=data_response["title"],
                prefix=data_response["prefix"],
                suffix=data_response["suffix"]
            ),
            address=ResumeLocation(
                formatted_location=data_response["formatted_location"],
                postal_code=data_response["postal_code"],
                region=data_response["region"],
                country=data_response["country"],
                country_code=data_response["country_code"],
                raw_input_location=data_response["raw_input_location"],
                street=data_response["street"],
                street_number=data_response["street_number"],
                appartment_number=data_response["appartment_number"],
                city=data_response["city"]
            ),
            self_summary=data_response["self_summary"],
            objective=data_response["objective"],
            date_of_birth=data_response["date_of_birth"],
            place_of_birth=data_response["place_of_birth"],
            phones=data_response["phones"],
            mails=data_response["mails"],
            urls=data_response["urls"],
            fax=data_response["fax"],
            current_profession=data_response["current_profession"],
            dateOfBirth=data_response["dateOfBirth"],
            gender=data_response["gender"],
            nationality=data_response["nationality"],
            martial_status=data_response["martial_status"],
            current_salary=data_response["current_salary"]
        )



        education_entries = []
        for entry in data_response["education"]["entries"]:
            education_entry = ResumeEducationEntry(
                title=entry["title"],
                start_date=entry["start_date"],
                end_date=entry["end_date"],
                location=ResumeLocation(
                    formatted_location=entry["location"]["formatted_location"],
                    postal_code=entry["location"]["postal_code"],
                    region=entry["location"]["region"],
                    country=entry["location"]["country"],
                    country_code=entry["location"]["country_code"],
                    raw_input_location=entry["location"]["raw_input_location"],
                    street=entry["location"]["street"],
                    street_number=entry["location"]["street_number"],
                    appartment_number=entry["location"]["appartment_number"],
                    city=entry["location"]["city"]
                ),
                establishment=entry["establishment"],
                description=entry["description"],
                gpa=entry["gpa"],
                accreditation=entry["accreditation"]
            )
            education_entries.append(education_entry)

        education_info = ResumeEducation(
            total_years_education=data_response["education"]["total_years_education"],
            entries=education_entries
        )

        work_exp_entries=[]
        for entry in data_response["work_experience"]["entries"]:
            work_exp_entry = ResumeWorkExpEntry(
                title=entry["title"],
                start_date=entry["start_date"],
                end_date=entry["end_date"],
                company=entry["company"],
                location=ResumeLocation(
                    formatted_location=entry["location"]["formatted_location"],
                    postal_code=entry["location"]["postal_code"],
                    region=entry["location"]["region"],
                    country=entry["location"]["country"],
                    country_code=entry["location"]["country_code"],
                    raw_input_location=entry["location"]["raw_input_location"],
                    street=entry["location"]["street"],
                    street_number=entry["location"]["street_number"],
                    appartment_number=entry["location"]["appartment_number"],
                    city=entry["location"]["city"]
                ),
                description=entry["description"],
                industry=entry["industry"]
            )
            work_exp_entries.append(work_exp_entry)
        work_experience = ResumeWorkExp(
            total_years_experience=data_response["work_experience"]["total_years_experience"],
            entries=work_exp_entries
        )

        language_info=[]
        for l in data_response["languages"]:
            language = ResumeLang(
                name=l["name"],
                code=l["code"]
            )
            language_info.append(language)

        skill_info = []
        for s in data_response["skills"]:
            skill = ResumeSkill(
                name=s["name"],
                type=s["type"]
            )
            skill_info.append(skill)

        certification_info = []
        for cert in data_response["certifications"]:
            cert = ResumeSkill(
                name=cert["name"],
                type=cert["type"]
            )
            certification_info.append(skill)

        course_info = []
        for co in data_response["courses"]:
            co = ResumeSkill(
                name=co["name"],
                type=co["type"]
            )
            course_info.append(skill)

        publication_info = []
        for p in data_response["publications"]:
            p = ResumeSkill(
                name=p["name"],
                type=p["type"]
            )
            publication_info.append(skill)

        interest_info = []
        for i in data_response["interests"]:
            i = ResumeSkill(
                name=i["name"],
                type=i["type"]
            )
            interest_info.append(skill)
        extracted_data = ResumeExtractedData(
            personal_infos=personal_info,
            education=education_info,
            work_experience=work_exp_info,
            languages=language_info,
            skills=skill_info,
            certifications=certification_info,
            courses=course_info,
            publications=publication_info,
            interests=interest_info
        )
        standardized_response = ResumeParserDataClass(
            extracted_data=extracted_data
        )
        return ResponseType[ResumeParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response
        )    
        """
    def ocr__identity_parser(
        self,
        file: str,
        file_url: str = ""
    ) -> ResponseType[IdentityParserDataClass]:
        file_ = open(file, "rb")
        original_response = self._make_post_request(file_)
        file_.close()

        items = []
        fields = {}
        country = {}

        for document in original_response.get("documents", []):
            fields = document["fields"]
            country["value"] = get_info_country(
                key=InfoCountry.ALPHA3,
                value=fields.get("CountryRegion", {}).get("content"),
            )
            country["confidence"] = fields.get("CountryRegion", {}).get(
                "confidence"
            )

        given_names = original_response.get("FirstName", {}).get("content", "").split(" ")
        final_given_names = []
        for given_name in given_names:
            final_given_names.append(
                ItemIdentityParserDataClass(
                    value=given_name,
                    confidence=original_response.get("FirstName", {}).get("confidence"),
                )
            )

        items.append(
            InfosIdentityParserDataClass(
                last_name=InfosIdentityParserDataClass.to_uppercase(

                    ItemIdentityParserDataClass(
                        value=fields.get("last_name", {}).get("content"),
                        confidence=fields.get("last_name", {}).get("confidence"),
                    )
                ),
                given_names=[InfosIdentityParserDataClass.to_title(given_name) for given_name in final_given_names],
                birth_place=InfosIdentityParserDataClass.to_title(
                   
                    ItemIdentityParserDataClass(
                        value=fields.get("birth_place", {}).get("content"),
                        confidence=fields.get("birth_place", {}).get("confidence"),
                    )
                ),
                birth_date=ItemIdentityParserDataClass(
                    value=format_date(fields.get("DateOfBirth", {}).get("value")),
                    confidence=fields.get("DateOfBirth", {}).get("confidence"),
                ),
                issuance_date=ItemIdentityParserDataClass(
                    value=format_date(fields.get("DateOfIssue", {}).get("value")),
                    confidence=fields.get("DateOfIssue", {}).get("confidence"),
                ),
                expire_date=ItemIdentityParserDataClass(
                    value=format_date(
                        fields.get("DateOfExpiration", {}).get("value")
                    ),
                    confidence=fields.get("DateOfExpiration", {}).get("confidence"),
                ),
                document_id=ItemIdentityParserDataClass(
                    value=fields.get("document_id", {}).get("content"),
                    confidence=fields.get("document_id", {}).get("confidence"),
                ),
                issuing_state=InfosIdentityParserDataClass.to_title(
                  
                    ItemIdentityParserDataClass(
                        value=fields.get("issuing_state", {}).get("content"),
                        confidence=fields.get("issuing_state", {}).get("confidence"),
                    )
                ),
                address=ItemIdentityParserDataClass(
                    value=fields.get("address", {}).get("content"),
                    confidence=fields.get("address", {}).get("confidence"),
                ),
                age=ItemIdentityParserDataClass(
                    value=fields.get("age", {}).get("content"),
                    confidence=fields.get("age", {}).get("confidence"),
                ),
                country=country,
                document_type=InfosIdentityParserDataClass.to_title(
                   
                    ItemIdentityParserDataClass(
                        value=fields.get("document_type", {}).get("content"),
                        confidence=fields.get("document_type", {}).get("confidence"),
                    )
                ),
                gender=ItemIdentityParserDataClass(
                    value=fields.get("gender", {}).get("content"),
                    confidence=fields.get("gender", {}).get("confidence"),
                ),
                image_id=[ItemIdentityParserDataClass(
                    value=image.get("id"),
                    confidence=image.get("confidence"),
                ) for image in fields.get("images", [])],
                image_signature=[ItemIdentityParserDataClass(
                    value=image.get("signature"),
                    confidence=image.get("confidence"),
                ) for image in fields.get("images", [])],
                mrz=ItemIdentityParserDataClass(
                    value=fields.get("mrz", {}).get("content"),
                    confidence=fields.get("mrz", {}).get("confidence"),
                ),
                nationality=InfosIdentityParserDataClass.to_title(
                    
                    ItemIdentityParserDataClass(
                        value=fields.get("nationality", {}).get("content"),
                        confidence=fields.get("nationality", {}).get("confidence"),
                    )
                )
            )
        )

        standardized_response = IdentityParserDataClass(extracted_data=items)
        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )



    


                


            

            
            
















