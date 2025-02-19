import json
import mimetypes
import uuid
from typing import Sequence

import google.auth
import googleapiclient.discovery
from PIL import Image as Img, UnidentifiedImageError
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import vision
from google.cloud.documentai_v1beta3 import Document
from google.cloud.vision_v1.types.image_annotator import (
    EntityAnnotation,
)
from google.protobuf.json_format import MessageToDict

from edenai_apis.apis.google.google_helpers import (
    google_ocr_tables_standardize_response,
    handle_done_response_ocr_async,
    handle_google_call,
    google_financial_parser,
)
from edenai_apis.features.ocr import (
    BankInvoice,
    CustomerInformationInvoice,
    InfosInvoiceParserDataClass,
    InvoiceParserDataClass,
    ItemLinesInvoice,
    LocaleInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
    Bounding_box,
    OcrDataClass,
    OcrTablesAsyncDataClass,
)
from edenai_apis.features.ocr.ocr_async.ocr_async_dataclass import (
    OcrAsyncDataClass,
)
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.ocr.financial_parser.financial_parser_dataclass import (
    FinancialParserDataClass,
    FinancialParserType,
)
from edenai_apis.features.ocr.receipt_parser.receipt_parser_dataclass import (
    InfosReceiptParserDataClass,
    ItemLines,
    Locale,
    MerchantInformation,
    PaymentInformation,
    ReceiptParserDataClass,
    Taxes,
)
from edenai_apis.utils.conversion import convert_string_to_number
from edenai_apis.utils.exception import (
    ProviderException,
)
from edenai_apis.utils.pdfs import get_pdf_width_height
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)


class GoogleOcrApi(OcrInterface):
    def ocr__ocr(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[OcrDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()

        image = vision.Image(content=file_content)
        payload = {"image": image}
        response = handle_google_call(self.clients["image"].text_detection, **payload)

        mimetype = mimetypes.guess_type(file)[0] or "unrecognized"
        if mimetype.startswith("image"):
            try:
                with Img.open(file) as img:
                    width, height = img.size
            except UnidentifiedImageError as exc:
                raise ProviderException(
                    "Image could not be identified. Supported types are: image/* and application/pdf"
                ) from exc
        elif mimetype == "application/pdf":
            width, height = get_pdf_width_height(file)
        else:
            raise ProviderException(
                "File type not supported by Google OCR API. Supported types are: image/* and application/pdf"
            )

        boxes: Sequence[Bounding_box] = []
        final_text = ""
        original_response = MessageToDict(response._pb)

        text_annotations: Sequence[EntityAnnotation] = response.text_annotations
        if text_annotations and isinstance(text_annotations[0], EntityAnnotation):
            final_text += text_annotations[0].description.replace("\n", " ")
        for text in text_annotations[1:]:
            xleft = float(text.bounding_poly.vertices[0].x)
            xright = float(text.bounding_poly.vertices[1].x)
            ytop = float(text.bounding_poly.vertices[0].y)
            ybottom = float(text.bounding_poly.vertices[2].y)
            boxes.append(
                Bounding_box(
                    text=text.description,
                    left=float(xleft / width),
                    top=float(ytop / height),
                    width=(xright - xleft) / width,
                    height=(ybottom - ytop) / height,
                )
            )
        standardized = OcrDataClass(
            text=final_text.replace("\n", " ").strip(), bounding_boxes=boxes
        )
        return ResponseType[OcrDataClass](
            original_response=original_response, standardized_response=standardized
        )

    def ocr__receipt_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[ReceiptParserDataClass]:
        mimetype = mimetypes.guess_type(file)[0] or "unrecognized"

        receipt_parser_project_id = self.api_settings["documentai"]["project_id"]
        receipt_parser_process_id = self.api_settings["documentai"][
            "process_receipt_id"
        ]

        opts = ClientOptions(api_endpoint=f"eu-documentai.googleapis.com")
        receipt_client = documentai.DocumentProcessorServiceClient(client_options=opts)
        name = receipt_client.processor_path(
            receipt_parser_project_id, "eu", receipt_parser_process_id
        )

        with open(file, "rb") as file_:
            raw_document = documentai.RawDocument(
                content=file_.read(), mime_type=mimetype
            )

            payload_request = {"name": name, "raw_document": raw_document}
            request = handle_google_call(documentai.ProcessRequest, **payload_request)

            payload_result = {"request": request}
            result = handle_google_call(
                receipt_client.process_document, **payload_result
            )

            document = result.document

        receipt_infos: InfosReceiptParserDataClass = InfosReceiptParserDataClass()
        receipt_taxe: Taxes = Taxes()
        item_lines: Sequence[ItemLines] = []
        local_receipt: Locale = Locale()
        payement_information: PaymentInformation = PaymentInformation()
        merchant_infos: MerchantInformation = MerchantInformation()

        entites: Sequence[Document.Entity] = document.entities

        item_lines_index = 0
        for entity in entites:
            entity_dict = Document.Entity.to_dict(entity)
            entity_type = entity_dict.get("type_", "")
            entity_value = entity_dict.get("normalized_value", {}).get(
                "text"
            ) or entity_dict.get("mention_text")
            if entity_type == "credit_card_last_four_digits":
                payement_information.card_number = entity_value
            if entity_type == "currency":
                local_receipt.currency = entity_value
            if entity_type == "start_date":
                receipt_infos.date = entity_value
            if entity_type == "end_date":
                receipt_infos.due_date = entity_value
            if entity_type == "net_amount":
                string_amount = entity_value
                amount = convert_string_to_number(string_amount, float)
                receipt_infos.invoice_subtotal = amount
            if entity_type == "purchase_time":
                receipt_infos.time = entity_value
            if entity_type == "receipt_date":
                receipt_infos.date = entity_value
            if entity_type == "supplier_address":
                merchant_infos.merchant_address = entity_value
            if entity_type == "supplier_name":
                merchant_infos.merchant_name = entity_value
            if entity_type == "supplier_phone":
                merchant_infos.merchant_phone = entity_value
            if entity_type == "tip_amount":
                payement_information.tip = entity_value
            if entity_type == "total_amount":
                string_amount = entity_value
                amount = convert_string_to_number(string_amount, float)
                receipt_infos.invoice_total = amount
            if entity_type == "total_tax_amount":
                string_amount = entity_value
                amount = convert_string_to_number(string_amount, float)
                receipt_taxe.taxes = amount
            if entity_type == "line_item":
                item = ItemLines()
                for property in entity_dict.get("properties", []) or []:
                    property_type = property.get("type_", "")
                    property_value = property.get("normalized_value", {}).get(
                        "text"
                    ) or property.get("mention_text")
                    if property_type == "line_item/amount":
                        string_amount = property_value
                        amount = convert_string_to_number(string_amount, float)
                        item.amount = amount
                    if property_type == "line_item/description":
                        item.description = property_value
                    if property_type == "line_item/quantity":
                        string_quantity = property_value
                        quantity = convert_string_to_number(string_quantity, float)
                        item.quantity = quantity
                item_lines.append(item)

            item_lines_index += 1

        receipt_infos.merchant_information = merchant_infos
        receipt_infos.taxes = [receipt_taxe]
        receipt_infos.item_lines = item_lines
        receipt_infos.locale = local_receipt
        receipt_infos.payment_information = payement_information

        return ResponseType[ReceiptParserDataClass](
            original_response=Document.to_dict(document),
            standardized_response=ReceiptParserDataClass(
                extracted_data=[receipt_infos]
            ),
        )

    def ocr__invoice_parser(
        self, file: str, language: str, file_url: str = "", **kwargs
    ) -> ResponseType[InvoiceParserDataClass]:
        mimetype = mimetypes.guess_type(file)[0] or "unrecognized"

        invoice_parser_project_id = self.api_settings["documentai"]["project_id"]
        invoice_parser_process_id = self.api_settings["documentai"][
            "process_invoice_id"
        ]

        opts = ClientOptions(api_endpoint=f"eu-documentai.googleapis.com")
        invoice_client = documentai.DocumentProcessorServiceClient(client_options=opts)
        name = invoice_client.processor_path(
            invoice_parser_project_id, "eu", invoice_parser_process_id
        )

        with open(file, "rb") as file_:
            raw_document = documentai.RawDocument(
                content=file_.read(), mime_type=mimetype
            )

        payload_request = {"name": name, "raw_document": raw_document}
        request = handle_google_call(documentai.ProcessRequest, **payload_request)

        payload_result = {"request": request}
        result = handle_google_call(invoice_client.process_document, **payload_result)

        document = result.document

        invoice_infos = []
        entites: Sequence[Document.Entity] = document.entities

        for idx in range(0, len(Document.to_dict(document).get("pages"))):
            invoice_infos.append(InfosInvoiceParserDataClass())
            merchant_infos: MerchantInformationInvoice = (
                MerchantInformationInvoice.default()
            )
            customer_infos: CustomerInformationInvoice = (
                CustomerInformationInvoice.default()
            )
            local_invoice: LocaleInvoice = LocaleInvoice.default()
            item_lines: Sequence[ItemLinesInvoice] = []
            bank_invoice: BankInvoice = BankInvoice.default()
            invoice_taxe: TaxesInvoice = TaxesInvoice.default()
            for entity in entites:
                entity_dict = Document.Entity.to_dict(entity)
                page_anchor = entity_dict.get("page_anchor", {}) or {}
                page_refs = page_anchor.get("page_refs", [{}]) or [{}]
                if page_refs[0].get("page") != str(idx):
                    continue

                entity_type = entity_dict.get("type_", "")
                entity_value = entity_dict.get("normalized_value", {}).get(
                    "text"
                ) or entity_dict.get("mention_text")
                if entity_type == "total_amount":
                    string_amount = entity_value
                    amount = convert_string_to_number(string_amount, float)
                    invoice_infos[idx].invoice_total = amount
                if entity_type == "net_amount":
                    string_amount = entity_value
                    amount = convert_string_to_number(string_amount, float)
                    invoice_infos[idx].invoice_subtotal = amount
                if entity_type == "total_tax_amount":
                    string_amount = entity_value
                    amount = convert_string_to_number(string_amount, float)
                    invoice_taxe.value = amount
                if entity_type == "currency_exchange_rate":
                    string_amount = entity_value
                    amount = convert_string_to_number(string_amount, float)
                    invoice_taxe.rate = amount
                if entity_type == "payment_terms":
                    invoice_infos[idx].payment_term = entity_value
                if entity_type == "supplier_iban":
                    bank_invoice.iban = entity_value
                if entity_type == "currency":
                    local_invoice.currency = entity_value
                if entity_type == "invoice_id":
                    invoice_infos[idx].invoice_number = entity_value
                if entity_type == "purchase_order":
                    invoice_infos[idx].purchase_order = entity_value
                if entity_type == "invoice_date":
                    invoice_infos[idx].date = entity_value
                if entity_type == "due_date":
                    invoice_infos[idx].due_date = entity_value
                if entity_type == "delivery_date":
                    invoice_infos[idx].service_date = entity_value
                if entity_type == "supplier_name":
                    merchant_infos.merchant_name = entity_value
                if entity_type == "supplier_email":
                    merchant_infos.merchant_address = entity_value
                if entity_type == "supplier_phone":
                    merchant_infos.merchant_phone = entity_value
                if entity_type == "supplier_address":
                    merchant_infos.merchant_address = entity_value
                if entity_type == "supplier_tax_id":
                    merchant_infos.merchant_tax_id = entity_value
                if entity_type == "supplier_website":
                    merchant_infos.merchant_website = entity_value
                if entity_type == "receiver_name":
                    customer_infos.customer_name = entity_value
                if entity_type == "receiver_address":
                    customer_infos.customer_address = entity_value
                if entity_type == "receiver_tax_id":
                    customer_infos.customer_tax_id = entity_value
                if entity_type == "receiver_email":
                    customer_infos.customer_email = entity_value
                if entity_type == "remit_to_address":
                    customer_infos.customer_remittance_address = entity_value
                if entity_type == "ship_to_address":
                    customer_infos.customer_shipping_address = entity_value

                if entity_type == "line_item":
                    item = ItemLinesInvoice()
                    for property in entity_dict.get("properties", []) or []:
                        property_type = property.get("type_", "")
                        property_value = property.get("normalized_value", {}).get(
                            "text"
                        ) or property.get("mention_text")
                        if property_type == "line_item/amount":
                            string_amount = property_value
                            amount = convert_string_to_number(string_amount, float)
                            item.amount = amount
                        if property_type == "line_item/quantity":
                            string_quantity = property_value
                            quantity = convert_string_to_number(string_quantity, float)
                            item.quantity = quantity
                        if property_type == "line_item/unit_price":
                            string_unit_price = property_value
                            unit_price = convert_string_to_number(
                                string_unit_price, float
                            )
                            item.unit_price = unit_price
                        if property_type == "line_item/description":
                            item.description = property_value
                        if property_type == "line_item/product_code":
                            item.product_code = property_value
                    item_lines.append(item)

            invoice_infos[idx].merchant_information = merchant_infos
            invoice_infos[idx].customer_information = customer_infos
            invoice_infos[idx].locale = local_invoice
            invoice_infos[idx].bank_informations = bank_invoice
            invoice_infos[idx].item_lines = item_lines
            invoice_infos[idx].taxes = [invoice_taxe]

        return ResponseType[InvoiceParserDataClass](
            original_response=Document.to_dict(document),
            standardized_response=InvoiceParserDataClass(extracted_data=invoice_infos),
        )

    def ocr__ocr_tables_async__launch_job(
        self, file: str, file_type: str, language: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        file_name: str = file.split("/")[-1]  # file.name give its whole path

        documentai_projectid = self.api_settings["documentai"]["project_id"]
        documentai_processid = self.api_settings["documentai"]["process_ocr_tables_id"]

        gcs_output_uri = "gs://async-ocr-tables"
        gcs_output_uri_prefix = "outputs"
        gcs_input_uri = f"gs://async-ocr-tables/{file_name}"

        # upload file to bucket
        bucket_client = self.clients["storage"]
        ocr_tables_bucket = bucket_client.get_bucket("async-ocr-tables")
        new_blob = ocr_tables_bucket.blob(file_name)
        new_blob.upload_from_filename(file)

        doc_ai = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": "eu-documentai.googleapis.com"}
        )

        destination_uri = f"{gcs_output_uri}/{gcs_output_uri_prefix}/"

        gcs_documents = documentai.GcsDocuments(
            documents=[{"gcs_uri": gcs_input_uri, "mime_type": file_type}]
        )

        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config={"gcs_uri": destination_uri}
        )

        name = f"projects/{documentai_projectid}/locations/eu/processors/{documentai_processid}"

        request = documentai.BatchProcessRequest(
            name=name,
            input_documents=input_config,
            document_output_config=output_config,
        )
        response = doc_ai.batch_process_documents(request)

        operation_id = response.operation.name.split("/")[-1]

        return AsyncLaunchJobResponseType(provider_job_id=operation_id)

    def ocr__ocr_tables_async__get_job_result(
        self, job_id: str
    ) -> ResponseType[OcrTablesAsyncDataClass]:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials, _ = google.auth.default(scopes=scopes)

        documentai_projectid = self.api_settings["documentai"]["project_id"]

        name = f"projects/{documentai_projectid}/locations/eu/operations/{job_id}"

        service = googleapiclient.discovery.build(
            serviceName="documentai",
            version="v1beta3",
            credentials=credentials,
            client_options={"api_endpoint": "https://eu-documentai.googleapis.com"},
        )

        request = service.projects().locations().operations().get(name=name)

        res = handle_google_call(request.execute)

        if res["metadata"]["state"] == "SUCCEEDED":
            # get result from bucket
            bucket_client = self.clients["storage"]
            ocr_tables_bucket = bucket_client.get_bucket("async-ocr-tables")
            output_uri = res["metadata"]["individualProcessStatuses"][0][
                "outputGcsDestination"
            ]
            prefix = output_uri.split("gs://async-ocr-tables/")[1] + "/"
            blob_list = ocr_tables_bucket.list_blobs(prefix=prefix)
            # convert byte array to json
            byte_res = list(blob_list)[0].download_as_bytes()
            original_result = json.loads(byte_res)

            standardized_response = google_ocr_tables_standardize_response(
                original_result
            )
            return AsyncResponseType[OcrTablesAsyncDataClass](
                status="succeeded",
                original_response=original_result,
                standardized_response=standardized_response,
                provider_job_id=job_id,
            )

        elif res["metadata"]["state"] == "FAILED":
            raise ProviderException(res.get("error"))

        return AsyncPendingResponseType[OcrTablesAsyncDataClass](
            status="pending", provider_job_id=job_id
        )

    def ocr__ocr_async__launch_job(
        self, file: str, file_url: str = "", **kwargs
    ) -> AsyncLaunchJobResponseType:
        call_uuid = uuid.uuid4().hex
        filename: str = call_uuid + file.split("/")[-1]

        gcs_output_uri = "gs://ocr-async/outputs-" + call_uuid
        gcs_input_uri = f"gs://ocr-async/{filename}"

        ocr_async_bucket = self.clients["storage"].get_bucket("ocr-async")
        new_blob = ocr_async_bucket.blob(filename)
        new_blob.upload_from_filename(file)

        feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

        mime_type, _ = mimetypes.guess_type(file)

        gcs_source = vision.GcsSource(uri=gcs_input_uri)
        input_config = vision.InputConfig(gcs_source=gcs_source, mime_type=mime_type)

        gcs_destination = vision.GcsDestination(uri=gcs_output_uri)
        output_config = vision.OutputConfig(
            gcs_destination=gcs_destination, batch_size=1
        )

        async_request = vision.AsyncAnnotateFileRequest(
            features=[feature], input_config=input_config, output_config=output_config
        )

        response = self.clients["image"].async_batch_annotate_files(
            requests=[async_request]
        )

        operation_id = response.operation.name.split("/")[-1]
        return AsyncLaunchJobResponseType(provider_job_id=operation_id)

    def ocr__ocr_async__get_job_result(
        self, job_id: str
    ) -> ResponseType[OcrAsyncDataClass]:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials, _ = google.auth.default(scopes=scopes)

        name = f"projects/{self.project_id}/operations/{job_id}"

        service = googleapiclient.discovery.build(
            serviceName="vision",
            version="v1",
            credentials=credentials,
            client_options={"api_endpoint": "https://vision.googleapis.com"},
        )

        request = service.projects().operations().get(name=name)

        res = handle_google_call(request.execute)

        if res["metadata"]["state"] == "DONE":
            return handle_done_response_ocr_async(res, self.clients["storage"], job_id)

        elif res["metadata"]["state"] == "FAILED":
            raise ProviderException(res.get("error"))

        return AsyncPendingResponseType[OcrTablesAsyncDataClass](provider_job_id=job_id)

    def ocr__financial_parser(
        self,
        file: str,
        language: str,
        document_type: str,
        file_url: str = "",
        model: str = None,
        **kwargs,
    ) -> ResponseType[FinancialParserDataClass]:
        mimetype = mimetypes.guess_type(file)[0] or "unrecognized"
        financial_project_id = self.api_settings["documentai"]["project_id"]

        document_type_key = (
            "process_invoice_id"
            if document_type == FinancialParserType.INVOICE.value
            else "process_receipt_id"
        )
        financial_parser_process_id = self.api_settings["documentai"][document_type_key]

        opts = ClientOptions(api_endpoint=f"eu-documentai.googleapis.com")
        financial_parser_client = documentai.DocumentProcessorServiceClient(
            client_options=opts
        )

        name = financial_parser_client.processor_path(
            financial_project_id, "eu", financial_parser_process_id
        )
        with open(file, "rb") as file_:
            raw_document = documentai.RawDocument(
                content=file_.read(), mime_type=mimetype
            )

            payload_request = {"name": name, "raw_document": raw_document}
            request = handle_google_call(documentai.ProcessRequest, **payload_request)

            payload_result = {"request": request}
            result = handle_google_call(
                financial_parser_client.process_document, **payload_result
            )
            document = result.document

        standardized_response = google_financial_parser(document)

        return ResponseType[FinancialParserDataClass](
            original_response=Document.to_dict(document),
            standardized_response=standardized_response,
        )
