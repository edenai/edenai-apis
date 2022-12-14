import functools
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories, FaceBoundingBox,
    FaceEmotions, FaceFacialHair,
    FaceHair, FaceHairColor, FaceItem,
    FaceLandmarks, FaceMakeup, FaceOcclusions,
    FacePoses, FaceQuality
)
from edenai_apis.features.ocr import (
    CustomerInformationInvoice,
    InfosInvoiceParserDataClass,
    InvoiceParserDataClass,
    ItemLinesInvoice,
    MerchantInformationInvoice,
    TaxesInvoice
)
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from edenai_apis.utils.conversion import combine_date_with_time


def get_microsoft_headers() -> Dict:
    api_settings = load_provider(ProviderDataEnum.KEY, "microsoft")
    return {
            "ocr_invoice": {
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": api_settings["ocr_invoice"][
                    "subscription_key"
                ],
            },
            "ocr_tables_async": {
                "Content-Type": "application/octet-stream",
                "Ocp-Apim-Subscription-Key": api_settings["ocr_tables_async"][
                    "subscription_key"
                ],
            },
            "vision": {
                "Ocp-Apim-Subscription-Key": api_settings["vision"][
                    "subscription_key"
                ],
                "Content-Type": "application/octet-stream",
            },
            "face": {
                "Ocp-Apim-Subscription-Key": api_settings["face"][
                    "subscription_key"
                ],
                "Content-Type": "application/octet-stream",
            },
            "text": {
                "Ocp-Apim-Subscription-Key": api_settings["text"][
                    "subscription_key"
                ],
            },
            "translator": {
                "Ocp-Apim-Subscription-Key": api_settings["translator"][
                    "subscription_key"
                ],
                "Ocp-Apim-Subscription-Region": api_settings["translator"][
                    "service_region"
                ],
                "Content-Type": "application/json",
            },
            "speech": {
                "Ocp-Apim-Subscription-Key": api_settings["speech"][
                    "subscription_key"
                ]
            },
        }

def get_microsoft_urls() -> Dict:
    api_settings = load_provider(ProviderDataEnum.KEY, "microsoft")
    return {
            "summarization": api_settings["summarization"]["url"],
            "ocr_invoice": api_settings["ocr_invoice"]["url"],
            "ocr_tables_async": api_settings["ocr_tables_async"]["url"],
            "vision": api_settings["vision"]["url"],
            "face": api_settings["face"]["url"],
            "text": api_settings["text"]["url"],
            "translator": api_settings["translator"]["url"],
            "speech": api_settings["speech"]["url"],
        }


def miscrosoft_normalize_face_detection_response(response, img_size):
    faces_list = []
    width, height = img_size
    for face in response:
        face_attr = face.get("faceAttributes", {})

        # features
        features = FaceMakeup(
            eye_make=face_attr.get("makeup", {}).get("eyeMakeup"),
            lip_make=face_attr.get("makeup", {}).get("lipMakeup"),
        )

        # accessories
        accessories = {}
        for access in face.get("accessories", []):
            if "type" in access:
                accessories[access["type"]] = access["confidence"]
        accessories_data_class = FaceAccessories(
            reading_glasses=accessories.get("glasses", 0.0),
            face_mask=accessories.get("mask", 0.0),
            headwear=accessories.get("headwear", 0.0),
        )

        # hair
        hair = face_attr.get("hair", {})
        face_color: Sequence[FaceHairColor] = []
        for color in hair.get("hairColor", []):
            face_color.append(
                FaceHairColor(
                    color=color.get("color"), confidence=color.get("confidence")
                )
            )

        hair_dataclass = FaceHair(
            hair_color=face_color,
            bald=face_attr.get("hair").get("bald", 0.0),
            invisible=face_attr.get("hair").get("invisible"),
        )

        facial_hair = FaceFacialHair(
            moustache=face_attr.get("facialHair", {}).get("moustache", None),
            beard=face_attr.get("facialHair", {}).get("beard", None),
            sideburns=face_attr.get("facialHair", {}).get("sideburns", None),
        )

        # quality
        quality_dataclass = FaceQuality(
            noise=face_attr["noise"].get("value", 0.0),
            exposure=face_attr["exposure"].get("value", 0.0),
            blur=face_attr["blur"].get("value", 0.0),
        )

        # occlusion
        face_occlusion = face_attr.get("occlusion", {})
        occlusions = FaceOcclusions(
            forehead_occluded=face_occlusion.get("foreheadOccluded", False),
            eye_occluded=face_occlusion.get("eyeOccluded", False),
            mouth_occluded=face_occlusion.get("foreheadOccluded", False),
        )

        # landmarks
        landmarks = {}
        for land in list(face.get("faceLandmarks").items()):
            landmarks[land[0]] = [land[1]["x"] / width, land[1]["y"] / height]

        face_landmarks = FaceLandmarks(
            left_eye_top=landmarks.get("eyeLeftTop", []),
            left_eye_right=landmarks.get("eyeLeftOuter", []),
            left_eye_bottom=landmarks.get("eyeLeftBottom", []),
            left_eye_left=landmarks.get("eyeLeftInner", []),
            right_eye_top=landmarks.get("eyeRightTop", []),
            right_eye_right=landmarks.get("eyeRightOuter", []),
            right_eye_bottom=landmarks.get("eyeRightBottom", []),
            right_eye_left=landmarks.get("eyeRightInner", []),
            left_eyebrow_left=landmarks.get("eyebrowLeftInner", []),
            left_eyebrow_right=landmarks.get("eyebrowLeftOuter", []),
            right_eyebrow_left=landmarks.get("eyebrowRightInner", []),
            right_eyebrow_right=landmarks.get("eyebrowRightOuter", []),
            left_pupil=landmarks.get("pupilLeft", []),
            right_pupil=landmarks.get("pupilRight", []),
            nose_tip=landmarks.get("noseTip", []),
            mouth_left=landmarks.get("MOUTH_LEFT", []),
            mouth_right=landmarks.get("MOUTH_RIGHT", []),
            nose_left_alar_out_tip=landmarks.get("noseLeftAlarOutTip", []),
            nose_left_alar_top=landmarks.get("noseLeftAlarTop", []),
            nose_right_alar_out_tip=landmarks.get("noseRightAlarOutTip", []),
            nose_right_alar_top=landmarks.get("noseRightAlarOutTip", []),
            nose_root_left=landmarks.get("noseRootLeft", []),
            nose_root_right=landmarks.get("noseRootRight", []),
            under_lip_bottom=landmarks.get("underLipBottom", []),
            under_lip_top=landmarks.get("underLipTop", []),
            upper_lip_bottom=landmarks.get("upperLipBottom", []),
            upper_lip_top=landmarks.get("upperLipTop", []),
        )

        # emotions
        emotions = FaceEmotions(
            joy=int(face_attr.get("emotion").get("happiness")),
            sorrow=int(face_attr.get("emotion").get("sadness")),
            anger=int(face_attr.get("emotion").get("anger")),
            surprise=int(face_attr.get("emotion").get("surprise")),
            disgust=int(face_attr.get("emotion").get("disgust")),
            contempt=int(face_attr.get("emotion").get("contempt")),
            fear=int(face_attr.get("emotion").get("fear")),
            neutral=int(face_attr.get("emotion").get("neutral")),
        )

        # poses
        poses = FacePoses(
            pitch=face_attr.get("headPose").get("pitch"),
            roll=face_attr.get("headPose").get("roll"),
            yaw=face_attr.get("headPose").get("yaw"),
        )

        # bounding boxes
        rect = face.get("faceRectangle", {})
        faces_list.append(
            FaceItem(
                age=face_attr.get("age"),
                gender=face_attr.get("gender"),
                hair=hair_dataclass,
                facial_hair=facial_hair,
                makeup=features,
                accessories=accessories_data_class,
                quality=quality_dataclass,
                emotions=emotions,
                landmarks=face_landmarks,
                poses=poses,
                occlusions=occlusions,
                confidence=1.0,  # HACK default confidence at 100% may not be the best solution
                bounding_box=FaceBoundingBox(
                    x_min=rect.get("left", 0) / width,
                    x_max=(rect.get("left", 0) + rect.get("width", 0)) / width,
                    y_min=rect.get("top", 0) / height,
                    y_max=(rect.get("top", 0) + rect.get("height", 0)) / height,
                ),
            )
        )
    return deepcopy(faces_list)


# _Transform score of confidence to level of confidence
def content_processing(confidence):

    if confidence is not None:
        confidence = confidence * 100
        processing = 0
        if confidence < 10:
            processing = 1
        elif confidence < 30:
            processing = 2
        elif confidence < 60:
            processing = 3
        elif confidence < 80:
            processing = 4
        elif confidence > 80:
            processing = 5
        else:
            processing = 0
    return processing


def normalize_invoice_result(response):
    """normalize the original response of the provider api"""

    invoice_data = response[0]["fields"]
    default_dict = defaultdict(lambda: None)
    customer_address = invoice_data.get("CustomerAddress", default_dict).get(
        "text", None
    )
    customer_name = invoice_data.get("CustomerName", default_dict).get(
        "text", None
    )
    merchant_address = invoice_data.get("VendorAddress", default_dict).get(
        "text", None
    )
    merchant_name = invoice_data.get("VendorName", default_dict).get(
        "text", None
    )
    invoice_total = invoice_data.get("InvoiceTotal", default_dict).get(
        "value", None
    )
    invoice_subtotal = invoice_data.get("SubTotal", default_dict).get(
        "value", None
    )
    invoice_number = (
        invoice_data.get("InvoiceId", default_dict)
        .get("value_data", default_dict)
        .get("text", None)
    )
    date = (
        invoice_data.get("InvoiceDate", default_dict)
        .get("value_data", default_dict)
        .get("text", None)
    )
    invoice_time = (
        invoice_data.get("InvoiceTime", default_dict)
        .get("value_data", default_dict)
        .get("text", None)
    )
    date = combine_date_with_time(date, invoice_time)
    due_date = (
        invoice_data.get("DueDate", default_dict)
        .get("value_data", default_dict)
        .get("text", None)
    )
    taxes = [
        TaxesInvoice(
            value=invoice_data.get("TotalTax", default_dict).get("value", None)
        )
    ]
    item_lines: Sequence[ItemLinesInvoice] = (
        [
            ItemLinesInvoice(
                amount=item.get("value", default_dict)
                .get("Amount", default_dict)
                .get("value", None),
                description=item.get("value", default_dict)
                .get("Description", default_dict)
                .get("value", None),
                quantity=int(
                    item.get("value", default_dict)
                    .get("Quantity", default_dict)
                    .get("value", None)
                ),
                unit_price=item.get("value", default_dict)
                .get("UnitPrice", default_dict)
                .get("value", None),
            )
            for item in invoice_data.get("Items", default_dict).get("value", [])
            if item.get("value_type", None) == "list"
        ]
        if invoice_data.get("Items", default_dict).get("value_type", None)
        == "list"
        else []
    )

    invoice_parser = InfosInvoiceParserDataClass(
        invoice_number=invoice_number,
        customer_information=CustomerInformationInvoice(
            customer_name=customer_name, customer_address=customer_address
        ),
        date=date,
        due_date=due_date,
        invoice_total=invoice_total,
        invoice_subtotal=invoice_subtotal,
        merchant_information=MerchantInformationInvoice(
            merchant_name=merchant_name,
            merchant_address=merchant_address
        ),
        taxes=taxes,
        item_lines=item_lines,
    )

    standarized_response = InvoiceParserDataClass(extracted_data=[invoice_parser])

    return standarized_response

def microsoft_ocr_tables_standardize_response(original_response) -> OcrTablesAsyncDataClass:
    num_pages = len(original_response['pages'])
    pages: List[Page] = [Page() for _ in range(num_pages)]

    for table in original_response.get("tables", []):
        std_table = _ocr_tables_standardize_table(table, original_response)
        page_num = table["boundingRegions"][0]["pageNumber"]
        pages[page_num-1].tables.append(std_table, original_response)

    return OcrTablesAsyncDataClass(
        pages=pages, num_pages=num_pages
    )


def _ocr_tables_standardize_table(table, original_response) -> Table:
    num_rows = max([cell['rowIndex'] for cell in table['cells']]) + 1
    rows = [Row() for _ in range(num_rows)]

    for cell in table["cells"]:
        std_cell = _ocr_tables_standardize_cell(cell, original_response)
        row = rows[cell['rowIndex']]
        row.cells.append(std_cell)
        row.is_header = False  # TODO set is_header

    std_table = Table(rows=rows, num_cols=table["columnCount"], num_rows=table["rowCount"])
    return std_table


def _ocr_tables_standardize_cell(cell, original_response) -> Cell:
    current_page_num = cell['boundingRegions'][0]['pageNumber']
    width = original_response["pages"][current_page_num - 1]["width"]
    height = original_response["pages"][current_page_num - 1]["height"]
    bounding_box = cell["boundingRegions"][0]["boundingBox"]

    return Cell(
        text=cell["content"],
        row_span=cell["rowSpan"],
        col_span=cell["columnSpan"],
        bounding_box=BoundixBoxOCRTable(
            height=(bounding_box[7] - bounding_box[3]) / height,
            width=(bounding_box[2] - bounding_box[0]) / width,
            left=bounding_box[1] / width,
            top=bounding_box[0] / height,
        ),
    )
