import json
import os
import fitz
from time import sleep

from edenai_apis.features.ocr import (
    FinancialParserDataClass,
    IdentityParserDataClass,
    ResumeParserDataClass,
)

from edenai_apis.utils.types import ResponseType
from edenai_apis.features import OcrInterface
from edenai_apis.utils.exception import ProviderException


def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except:
        raise FileNotFoundError
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    if doc:
        doc.close()
    return text


class OpenaiDocParsingApi(OcrInterface):

    def __assistant_parser(
        self,
        name,
        instruction,
        message_text,
        example_file,
        input_file,
        dataclass,
        model,
    ):

        with open(os.path.join(os.path.dirname(__file__), example_file), "r") as f:
            output_response = json.load(f)["standardized_response"]

        assistant = self.client.beta.assistants.create(
            response_format={"type": "json_object"},
            name=name,
            instructions="{} You return a json output and nothing else than a json output. The json should be shaped like the following with the exact same structure and the exact same keys but change the values to extract the inputed document informations : \n {}  \n\n The json output should follow this pydantic schema \n {} \n\n Your response should directly start with '{{' ".format(
                instruction, output_response, dataclass.schema()
            ),
            model=model,
        )

        input_file_text = extract_text_from_pdf(input_file)

        thread = self.client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": message_text + input_file_text,
                }
            ]
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        while run.status != "completed":
            sleep(1)

        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        usage = run.to_dict()["usage"]
        original_response = messages.to_dict()
        original_response["usage"] = usage

        try:
            standardized_response = json.loads(
                json.loads(messages.data[0].content[0].json())["text"]["value"]
            )
        except json.JSONDecodeError as exc:
            raise ProviderException(
                "An error occurred while parsing the response."
            ) from exc

        return original_response, standardized_response

    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str = "",
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:

        original_response, result = self.__assistant_parser(
            name="Invoice Parser",
            instruction="You are an invoice parsing model. You take an invoice as an input and extract information from it.",
            message_text="Analyse this invoice : \n",
            example_file="outputs/ocr/financial_parser_output.json",
            input_file=file,
            dataclass=FinancialParserDataClass,
            model=model,
        )

        return ResponseType[FinancialParserDataClass](
            original_response=original_response, standardized_response=result
        )

    def ocr__resume_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[ResumeParserDataClass]:

        original_response, result = self.__assistant_parser(
            name="Resume Parser",
            instruction="You are an resume parsing model.",
            message_text="Analyse this resume :",
            example_file="outputs/ocr/resume_parser_output.json",
            input_file=file,
            dataclass=ResumeParserDataClass,
            model=model,
        )

        return ResponseType[ResumeParserDataClass](
            original_response=original_response,
            standardized_response=result,
        )

    def ocr__identity_parser(
        self, file: str, file_url: str = "", model: str = None, **kwargs
    ) -> ResponseType[IdentityParserDataClass]:

        original_response, result = self.__assistant_parser(
            name="ID Parser",
            instruction="You are an Identy Documents parsing model.",
            message_text="Analyse this ID Document :",
            example_file="outputs/ocr/identity_parser_output.json",
            input_file=file,
            dataclass=IdentityParserDataClass,
            model=model,
        )

        return ResponseType[IdentityParserDataClass](
            original_response=original_response,
            standardized_response=result,
        )
