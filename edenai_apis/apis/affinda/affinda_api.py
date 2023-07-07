from io import BufferedReader
from typing import Dict, List, Sequence
from collections import defaultdict
from affinda import AffindaAPI, TokenCredential
from edenai_apis.features import OcrInterface
from edenai_apis.features.ocr import (
    ResumeEducationEntry,
    ResumeExtractedData,
    ResumeLang,
    ResumeParserDataClass,
    ResumePersonalInfo,
    ResumePersonalName,
    ResumeLocation,
    ResumeSkill,
    ResumeWorkExp,
    ResumeWorkExpEntry,
    InfosInvoiceParserDataClass,
    CustomerInformationInvoice,
    InvoiceParserDataClass,
    MerchantInformationInvoice,
    TaxesInvoice,
    BankInvoice,
    ItemLinesInvoice,
)
from edenai_apis.features.ocr.resume_parser.resume_parser_dataclass import (
    ResumeEducation,
)

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features.provider.provider_interface import ProviderInterface
from edenai_apis.utils.conversion import (
    combine_date_with_time,
    convert_string_to_number,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class AffindaApi(ProviderInterface, OcrInterface):
    provider_name = "affinda"

    def __init__(self, api_keys: Dict = {}):
        super().__init__()
        self.api_settings = load_provider(
            ProviderDataEnum.KEY, self.provider_name, api_keys=api_keys
        )
        credentials = TokenCredential(token=self.api_settings["api_key"])
        self.client = AffindaAPI(credential=credentials)

    def ocr__resume_parser(
        self, file: str, file_url: str = ""
    ) -> ResponseType[ResumeParserDataClass]:
        if file_url:
            original_response = self.client.create_document(
                url=file_url, workspace=self.api_settings["resume_workspace"]
            ).as_dict()
        else:
            file_ = open(file, "rb")
            original_response = self.client.create_document(
                file=file_, workspace=self.api_settings["resume_workspace"]
            ).as_dict()
            file_.close()

        if "detail" in original_response:
            raise ProviderException(original_response["detail"])

        if "errors" in original_response:
            raise ProviderException(original_response["errors"][0]["detail"])

        resume = original_response["data"]
        # 1. Personal informations
        # 1.1 Name
        name = resume.get("name", {})
        names = ResumePersonalName(
            raw_name=name.get("raw", ""),
            first_name=name.get("first", ""),
            last_name=name.get("last", ""),
            middle=name.get("middle"),
            title=name.get("title"),
            sufix=None,
            prefix=None,
        )

        # 1.2 Address
        location = resume.get("location", {})
        address = ResumeLocation(
            raw_input=location.get("rawInput"),
            postal_code=location.get("postalCode"),
            region=location.get("state"),
            country_code=location.get("countryCode"),
            country=location.get("country"),
            appartment_number=location.get("apartmentNumber"),
            city=location.get("city", ""),
            street=location.get("street"),
            street_number=location.get("streetNumber"),
            formatted_location=None,
            raw_input_location=None,
        )

        # 1.3 Others
        personal_infos = ResumePersonalInfo(
            name=names,
            address=address,
            phones=resume.get("phone_numbers"),
            mails=resume.get("emails"),
            urls=resume.get("websites"),
            self_summary=resume.get("summary"),
            current_profession=resume.get("profession"),
            objective=resume.get("objective"),
            date_of_birth=resume.get("dateOfBirth"),
            place_of_birth=None,
            gender=None,
            nationality=None,
            martial_status=None,
            current_salary=None,
        )

        # 2. Education
        edu_entries: List[ResumeEducationEntry] = []
        for i in resume["education"]:
            location = i.get("location", {})
            address = ResumeLocation(
                raw_input=location.get("rawInput"),
                postal_code=location.get("postalCode"),
                region=location.get("state"),
                country=location.get("country"),
                country_code=location.get("countryCode"),
                street_number=location.get("streetNumber"),
                street=location.get("street"),
                appartment_number=location.get("appartmentNumber"),
                city=location.get("city", ""),
                formatted_location=None,
                raw_input_location=None,
            )
            dates = i.get("dates", {})
            edu_entries.append(
                ResumeEducationEntry(
                    location=address,
                    start_date=dates.get("start_date"),
                    end_date=dates.get("end_date"),
                    establishment=i.get("organization"),
                    gpa=i.get("grade", {}).get("value"),
                    title=None,
                    description=None,
                    accreditation=None,
                )
            )

        edu = ResumeEducation(entries=edu_entries, total_years_education=None)

        # Work experience
        work_entries = []
        for i in resume["work_experience"]:
            dates = i.get("dates", {})
            location = i.get("location", {})
            address = ResumeLocation(
                raw_input=location.get("rawInput"),
                postal_code=location.get("postalCode"),
                region=location.get("state"),
                country=location.get("country"),
                country_code=location.get("countryCode"),
                street_number=location.get("streetNumber"),
                street=location.get("street"),
                appartment_number=location.get("appartmentNumber"),
                city=location.get("city", ""),
                formatted_location=None,
                raw_input_location=None,
            )
            work_entries.append(
                ResumeWorkExpEntry(
                    title=i.get("job_title"),
                    company=i.get("organization"),
                    start_date=dates.get("start_date"),
                    end_date=dates.get("end_date"),
                    description=i.get("job_description"),
                    location=address,
                    industry=None,
                )
            )
        duration = resume.get("total_years_experience")
        work = ResumeWorkExp(total_years_experience=str(duration), entries=work_entries)

        # Others
        skills = []
        for i in resume.get("skills"):
            skill = i.get("name")
            skill_type = i.get("type").split("_skill")[0]
            skills.append(ResumeSkill(name=skill, type=skill_type))

        languages = [ResumeLang(name=i, code=None) for i in resume.get("languages", [])]
        certifications = [ResumeSkill(name=i) for i in resume.get("certifications", [])]
        publications = [ResumeSkill(name=i) for i in resume.get("publications", [])]
        std = ResumeParserDataClass(
            extracted_data=ResumeExtractedData(
                personal_infos=personal_infos,
                education=edu,
                work_experience=work,
                languages=languages,
                skills=skills,
                certifications=certifications,
                publications=publications,
            )
        )

        result = ResponseType[ResumeParserDataClass](
            original_response=original_response, standardized_response=std
        )
        return result

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = ""
    ) -> ResponseType[InvoiceParserDataClass]:
        if file_url:
            original_response = self.client.create_document(
                url=file_url, workspace=self.api_settings["invoice_workspace"]
            ).as_dict()
        else:
            file_ = open(file, "rb")
            original_response = self.client.create_document(
                file=file_, workspace=self.api_settings["invoice_workspace"]
            ).as_dict()
            file_.close()

        if "detail" in original_response:
            raise ProviderException(original_response["detail"])

        if "errors" in original_response:
            raise ProviderException(original_response["errors"][0]["detail"])

        invoice_data = original_response["data"]
        if invoice_data.get("tables"):
            del invoice_data["tables"]
        default_dict = defaultdict(lambda: None)
        # ------------------------------------------------------------#
        merchant_name = invoice_data.get("supplier_company_name", default_dict).get(
            "raw"
        )
        merchant_address = invoice_data.get("supplier_address", default_dict).get("raw")
        merchant_phone = invoice_data.get("supplier_phone_number", default_dict).get(
            "raw"
        )
        merchant_tax_id = invoice_data.get(
            "supplier_business_number", default_dict
        ).get("raw")
        merchant_email = invoice_data.get("supplier_email", default_dict).get("raw")
        merchant_fax = invoice_data.get("supplier_fax", default_dict).get("raw")
        merchant_website = invoice_data.get("supplier_website", default_dict).get("raw")
        # ------------------------------------------------------------#
        customer_name = invoice_data.get("customer_company_name", default_dict).get(
            "raw"
        )
        customer_id = invoice_data.get("customer_number", default_dict).get("raw")
        customer_billing_address = invoice_data.get(
            "customer_billing_address", default_dict
        ).get("raw")
        customer_shipping_address = invoice_data.get(
            "customer_delivery_address", default_dict
        ).get("raw")
        # ------------------------------------------------------------#
        invoice_number = invoice_data.get("invoice_number", default_dict).get(
            "raw", None
        )
        invoice_total = convert_string_to_number(
            invoice_data.get("payment_amount_total", default_dict).get("raw", None),
            float,
        )
        invoice_subtotal = convert_string_to_number(
            invoice_data.get("payment_amount_base", default_dict).get("parsed", None),
            float,
        )
        payment_term = invoice_data.get("payment_reference", default_dict).get("raw")
        amount_due = invoice_data.get("payment_amount_due", default_dict).get("raw")
        amount_due = convert_string_to_number(amount_due, float)
        purchase_order = invoice_data.get(
            "invoice_purchase_order_number", default_dict
        ).get("raw")
        # ------------------------------------------------------------#
        date = invoice_data.get("invoice_date", default_dict).get("raw", None)
        time = invoice_data.get("invoice_time", default_dict).get("raw", None)
        date = combine_date_with_time(date, time)
        due_date = invoice_data.get("payment_date_due", default_dict).get("raw", None)
        due_time = invoice_data.get("payment_time_due", default_dict).get("raw", None)
        due_date = combine_date_with_time(due_date, due_time)
        # ------------------------------------------------------------#
        taxes = convert_string_to_number(
            invoice_data.get("payment_amount_tax", default_dict).get("parsed", None),
            float,
        )
        iban = invoice_data.get("bank_iban", default_dict).get("raw")
        swift = invoice_data.get("bank_swift", default_dict).get("raw")
        bsb = invoice_data.get("bank_bsb", default_dict).get("raw")
        sort_code = invoice_data.get("bank_sort_code", default_dict).get("raw")
        account_number = invoice_data.get("bank_account_number", default_dict).get(
            "raw"
        )
        bank = BankInvoice(
            iban=iban,
            swift=swift,
            bsb=bsb,
            sort_code=sort_code,
            account_number=account_number,
            vat_number=None,
            rooting_number=None,
        )
        # ------------------------------------------------------------#
        tables = invoice_data.get("tables", [])
        item_lines: Sequence[ItemLinesInvoice] = []
        if tables:
            items = (
                tables[0].get("rows", [default_dict]) if tables[0] else [default_dict]
            )
            for line in items:
                item_lines.append(
                    ItemLinesInvoice(
                        unit_price=line.get("unit_price"),
                        quantity=line.get("quantity"),
                        tax_item=line.get("tax_total"),
                        amount=line.get("base_total"),
                        date_item=line.get("date"),
                        description=line.get("description"),
                        product_code=line.get("code"),
                    )
                )
        invoice_parser = InfosInvoiceParserDataClass(
            invoice_number=invoice_number,
            customer_information=CustomerInformationInvoice(
                customer_name=customer_name,
                customer_address=customer_billing_address,
                customer_billing_address=customer_billing_address,
                customer_id=customer_id,
                customer_shipping_address=customer_shipping_address,
                customer_email=None,
                customer_tax_id=None,
                customer_mailing_address=None,
                customer_remittance_address=None,
                customer_service_address=None,
                abn_number=None,
                gst_number=None,
                pan_number=None,
                vat_number=None,
            ),
            merchant_information=MerchantInformationInvoice(
                merchant_name=merchant_name,
                merchant_address=merchant_address,
                merchant_phone=merchant_phone,
                merchant_tax_id=merchant_tax_id,
                merchant_email=merchant_email,
                merchant_fax=merchant_fax,
                merchant_website=merchant_website,
                merchant_siren=None,
                merchant_siret=None,
                abn_number=None,
                gst_number=None,
                pan_number=None,
                vat_number=None,
            ),
            date=date,
            due_date=due_date,
            invoice_subtotal=invoice_subtotal,
            invoice_total=invoice_total,
            item_lines=item_lines,
            payment_term=payment_term,
            amount_due=amount_due,
            purchase_order=purchase_order,
            bank_informations=bank,
            taxes=[TaxesInvoice(value=taxes, rate=None)],
        )

        standardized_response = InvoiceParserDataClass(extracted_data=[invoice_parser])

        result = ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standardized_response=standardized_response,
        )
        return result
