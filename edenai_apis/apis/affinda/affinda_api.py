from io import BufferedReader
from collections import defaultdict
from affinda import AffindaAPI, TokenCredential
from edenai_apis.features import Ocr
from edenai_apis.features.ocr import (
    ResumeEducationEntry,
    ResumeExtractedData,
    ResumeLang,
    ResumeParserDataClass,
    ResumePersonalInfo,
    ResumeSkill,
    ResumeWorkExp,
    ResumeWorkExpEntry,
    InfosInvoiceParserDataClass,
    CustomerInformationInvoice,
    InvoiceParserDataClass,
    MerchantInformationInvoice,
    TaxesInvoice,
)

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.features.base_provider.provider_api import ProviderApi
from edenai_apis.utils.conversion import combine_date_with_time, convert_string_to_number
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import ResponseType


class AffindaApi(ProviderApi, Ocr):
    provider_name = "affinda"

    def __init__(self):
        super().__init__()
        self.api_settings = load_provider(ProviderDataEnum.KEY, self.provider_name)
        credentials = TokenCredential(token=self.api_settings["api_key"])
        self.client = AffindaAPI(credential=credentials)

    def ocr__resume_parser(
        self, file: BufferedReader
    ) -> ResponseType[ResumeParserDataClass]:
        original_response = self.client.create_resume(file=file).as_dict()

        if "detail" in original_response:
            raise ProviderException(original_response["detail"])

        resume = original_response["data"]

        personal_infos = ResumePersonalInfo(
            first_name=resume["name"].get("first"),
            last_name=resume["name"].get("last"),
            address=resume.get("location", {}).get("formatted"),
            phones=resume.get("phone_numbers"),
            mails=resume.get("emails"),
            urls=resume.get("websites"),
            self_summary=resume.get("summary"),
            current_profession=resume.get("profession"),
        )

        # Education
        edu_entries = []
        for i in resume["education"]:
            dates = i.get("dates", {})
            edu_entries.append(
                ResumeEducationEntry(
                    start_date=dates.get("start_date"),
                    end_date=dates.get("end_date"),
                    establishment=i.get("organization"),
                )
            )
        edu = ResumeEducationEntry(entries=edu_entries)

        # Work experience
        work_entries = []
        for i in resume["work_experience"]:
            dates = i.get("dates", {})
            work_entries.append(
                ResumeWorkExpEntry(
                    title=i.get("job_title"),
                    company=i.get("organization"),
                    start_date=dates.get("start_date"),
                    end_date=dates.get("end_date"),
                    description=i.get("job_description"),
                    location=i.get("location", {}).get("formatted"),
                )
            )
        duration = resume.get("total_years_experience")
        work = ResumeWorkExp(total_years_experience=duration, entries=work_entries)

        # Others
        skills = []
        for i in resume.get("skills"):
            skill = i.get("name")
            skill_type = i.get("type").split("_skill")[0]
            skills.append(ResumeSkill(name=skill, type=skill_type))

        languages = [ResumeLang(name=i) for i in resume.get("languages", [])]
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
            original_response=original_response, standarized_response=std
        )
        return result

    def ocr__invoice_parser(
        self, file: BufferedReader, language: str
    ) -> ResponseType[InvoiceParserDataClass]:

        original_response = self.client.create_invoice(file=file).as_dict()

        if "detail" in original_response:
            raise ProviderException(original_response["detail"])

        invoice_data = original_response["data"]
        if invoice_data.get("tables"):
            del invoice_data["tables"]
        default_dict = defaultdict(lambda: None)
        customer_name = invoice_data.get("customer_company_name", default_dict).get(
            "raw", None
        )
        customer_address = invoice_data.get(
            "customer_billing_address", default_dict
        ).get("raw", None)
        merchant_name = invoice_data.get("supplier_company_name", default_dict).get(
            "raw", None
        )
        merchant_address = invoice_data.get("supplier_address", default_dict).get(
            "raw", None
        )
        invoice_total = convert_string_to_number(
            invoice_data.get("payment_amount_total", default_dict).get("raw", None),
            float,
        )
        date = invoice_data.get("invoice_date", default_dict).get("raw", None)
        time = invoice_data.get("invoice_time", default_dict).get("raw", None)
        date = combine_date_with_time(date, time)
        due_date = invoice_data.get("payment_date_due", default_dict).get("raw", None)
        due_time = invoice_data.get("payment_time_due", default_dict).get("raw", None)
        due_date = combine_date_with_time(due_date, due_time)
        invoice_number = invoice_data.get("invoice_number", default_dict).get(
            "raw", None
        )
        taxes = convert_string_to_number(
            invoice_data.get("payment_amount_tax", default_dict).get("parsed", None),
            float,
        )
        invoice_subtotal = convert_string_to_number(
            invoice_data.get("payment_amount_base", default_dict).get("parsed", None),
            float,
        )

        invoice_parser = InfosInvoiceParserDataClass(
            invoice_number=invoice_number,
            customer_information=CustomerInformationInvoice(
                customer_name=customer_name, customer_address=customer_address
            ),
            merchant_information=MerchantInformationInvoice(
                merchant_name=merchant_name, merchant_address=merchant_address
            ),
            date=date,
            due_date=due_date,
            invoice_subtotal=invoice_subtotal,
            invoice_total=invoice_total,
            taxes=[TaxesInvoice(value=taxes)],
        )

        standarized_response = InvoiceParserDataClass(
            extracted_data=[invoice_parser]
        )

        result = ResponseType[InvoiceParserDataClass](
            original_response=original_response,
            standarized_response=standarized_response
        )
        return result
