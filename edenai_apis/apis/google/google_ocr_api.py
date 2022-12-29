import json
import mimetypes
from io import BufferedReader
from typing import Sequence

import googleapiclient.discovery
from edenai_apis.apis.google.google_helpers import ocr_tables_async_response_add_rows
from edenai_apis.features.ocr.ocr.ocr_dataclass import Bounding_box, OcrDataClass
from edenai_apis.features.ocr.ocr_interface import OcrInterface
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.pdfs import get_pdf_width_height
from edenai_apis.utils.types import (
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from PIL import Image as Img

import google.auth
from google.cloud import documentai_v1beta3 as documentai
from google.cloud import vision
from google.cloud.vision_v1.types.image_annotator import (
    AnnotateImageResponse,
    EntityAnnotation,
)


class GoogleOcrApi(OcrInterface):
    def ocr__ocr(
        self, file: BufferedReader, language: str
    ) -> ResponseType[OcrDataClass]:
        file_content = file.read()

        image = vision.Image(content=file_content)
        response = self.clients["image"].text_detection(image=image)

        mimetype = mimetypes.guess_type(file.name)[0] or "unrecognized"
        if mimetype.startswith("image"):
            width, height = Img.open(file).size
        elif mimetype == "application/pdf":
            width, height = get_pdf_width_height(file)

        messages_list = []
        boxes: Sequence[Bounding_box] = []
        final_text = ""
        image_response: AnnotateImageResponse = response
        # TO DO better original_response
        messages_list.append(image_response)

        text_annotations: Sequence[EntityAnnotation] = image_response.text_annotations
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
            original_response=messages_list, standardized_response=standardized
        )

    def ocr__ocr_tables_async__launch_job(
        self, file: BufferedReader, file_type: str, language: str
    ) -> AsyncLaunchJobResponseType:
        file_name: str = file.name.split("/")[-1]  # file.name give its whole path

        documentai_projectid = self.api_settings["documentai"]["project_id"]
        documentai_processid = self.api_settings["documentai"]["process_id"]

        gcs_output_uri = "gs://async-ocr-tables"
        gcs_output_uri_prefix = "outputs"
        gcs_input_uri = f"gs://async-ocr-tables/{file_name}"

        # upload file to bucket
        bucket_client = self.clients["storage"]
        ocr_tables_bucket = bucket_client.get_bucket("async-ocr-tables")
        new_blob = ocr_tables_bucket.blob(file_name)
        new_blob.upload_from_string(file.read())

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
        res = request.execute()

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
            byte_res = list(blob_list)[0].download_as_string()
            original_result = json.loads(byte_res.decode("utf8"))

            raw_text = original_result["text"]
            pages: Sequence[Page] = []
            num_pages = len(original_result["pages"])
            for page in original_result["pages"]:
                tables: Sequence[Table] = []
                if "tables" in page.keys():
                    for table in page["tables"]:
                        ocr_num_rows = 0
                        ocr_num_cols = 0
                        rows: Sequence[Row] = []
                        if "headerRows" in table.keys():
                            for row in table["headerRows"]:
                                ocr_num_rows += 1
                                row, num_row_cols = ocr_tables_async_response_add_rows(
                                    row, raw_text, is_header=True
                                )
                                ocr_num_cols = max(ocr_num_cols, num_row_cols)
                                rows.append(row)
                        if "bodyRows" in table.keys():
                            for row in table["bodyRows"]:
                                ocr_num_rows += 1
                                row, num_row_cols = ocr_tables_async_response_add_rows(
                                    row, raw_text
                                )
                                ocr_num_cols = max(ocr_num_cols, num_row_cols)
                                rows.append(row)
                        ocr_table = Table(
                            rows=rows, num_rows=ocr_num_rows, num_cols=ocr_num_cols
                        )
                        tables.append(ocr_table)
                    ocr_page = Page(tables=tables)
                    pages.append(ocr_page)
            standardized_response = OcrTablesAsyncDataClass(
                pages=pages, num_pages=num_pages
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
