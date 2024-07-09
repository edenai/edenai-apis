import json
import os
from time import sleep

from edenai_apis.features.ocr import (
            FinancialParserDataClass,
            IdentityParserDataClass,
            ResumeParserDataClass
            )

from edenai_apis.utils.types import ResponseType
from edenai_apis.features import OcrInterface


class OpenaiDocParsingApi(OcrInterface):

    def __assistant_parser(
            self,
            name,
            instruction,
            message_text,
            example_file,
            input_file,
            dataclass
            ):

        with open(os.path.join(os.path.dirname(__file__), example_file), "r") as f:
                output_response = json.load(f)["standardized_response"]

        assistant = self.client.beta.assistants.create(
            name=name,
            instructions="{} You return a json output shaped like the following with the exact same structure and the exact same keys : \n {}  \n\n The json output should follow this pydantic schema \n {}".format(instruction, output_response, dataclass.schema()),
            response_format={ "type": "json_object" },
            model="gpt-4o",
            )

        message_file = self.client.files.create(
                  file=open(input_file, "rb"), purpose="assistants"
                  )

        thread = self.client.beta.threads.create(
          messages=[
            {
              "role": "user",
              "content": message_text,
              "attachments": [
                { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
              ],
            }
          ]
        )

        run = self.client.beta.threads.runs.create_and_poll(
          thread_id=thread.id,
          assistant_id=assistant.id,
        )

        while run.status != 'completed':
          sleep(1)

        messages = self.client.beta.threads.messages.list(
            thread_id=thread.id
          )

        return json.loads(json.loads(messages.data[0].content[0].json())["text"]["value"]) 

    def ocr__financial_parser(
        self, file: str, language: str, document_type: str = "", file_url: str = ""
    ) -> ResponseType[FinancialParserDataClass]:

        result = self.__assistant_parser(
                name="Invoice Parser",
                instruction="You are an invoice parsing model.",
                message_text="Analyse this invoice :",
                example_file="outputs/ocr/financial_parser_output.json",
                input_file=file,
                dataclass=FinancialParserDataClass
                )

        return ResponseType[FinancialParserDataClass](
            original_response=result,
            standardized_response=result
        )


    def ocr__resume_parser(
        self, file: str, file_url: str = ""
    ) -> ResponseType[ResumeParserDataClass]:

        result = self.__assistant_parser(
                name="Resume Parser",
                instruction="You are an resume parsing model.",
                message_text="Analyse this resume :",
                example_file="outputs/ocr/resume_parser_output.json",
                input_file=file,
                dataclass=ResumeParserDataClass
                )


        return ResponseType[ResumeParserDataClass](
            original_response=result,
            standardized_response=result,
        )


    def ocr__identity_parser(
        self, file: str, file_url: str = ""
    ) -> ResponseType[IdentityParserDataClass]:

        result = self.__assistant_parser(
                name="ID Parser",
                instruction="You are an Identy Documents parsing model.",
                message_text="Analyse this ID Document :",
                example_file="outputs/ocr/identity_parser_output.json",
                input_file=file,
                dataclass=IdentityParserDataClass
                )

        return ResponseType[IdentityParserDataClass](
            original_response=result,
            standardized_response=result,
        )

