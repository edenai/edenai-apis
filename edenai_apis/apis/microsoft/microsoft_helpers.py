import datetime
from collections import defaultdict
from copy import deepcopy
from math import floor
from typing import Dict, List, Sequence

from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceBoundingBox,
    FaceEmotions,
    FaceFacialHair,
    FaceFeatures,
    FaceHair,
    FaceHairColor,
    FaceItem,
    FaceLandmarks,
    FaceMakeup,
    FaceOcclusions,
    FacePoses,
    FaceQuality,
)
from edenai_apis.features.ocr import (
    CustomerInformationInvoice,
    InfosInvoiceParserDataClass,
    InvoiceParserDataClass,
    ItemLinesInvoice,
    MerchantInformationInvoice,
    TaxesInvoice,
)
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
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import (
    format_date,
)
from edenai_apis.features.ocr.ocr_async.ocr_async_dataclass import (
    BoundingBox,
    Line,
    OcrAsyncDataClass,
    Word,
    Page as OcrAsyncPage,
)
from edenai_apis.features.ocr.ocr_tables_async.ocr_tables_async_dataclass import (
    BoundixBoxOCRTable,
    Cell,
    OcrTablesAsyncDataClass,
    Page,
    Row,
    Table,
)
from edenai_apis.features.text import (
    ModerationDataClass,
    TextModerationCategoriesMicrosoftEnum,
    TextModerationItem,
)
from edenai_apis.features.text.moderation.category import CategoryType
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.conversion import (
    combine_date_with_time,
    standardized_confidence_score,
    convert_time_to_string,
)
from edenai_apis.utils.ssml import convert_audio_attr_in_prosody_tag


def get_microsoft_headers() -> Dict:
    api_settings = load_provider(ProviderDataEnum.KEY, "microsoft")
    return {
        "vision": {
            "Ocp-Apim-Subscription-Key": api_settings["vision"]["subscription_key"],
            "Content-Type": "application/octet-stream",
        },
        "face": {
            "Ocp-Apim-Subscription-Key": api_settings["face"]["subscription_key"],
            "Content-Type": "application/octet-stream",
        },
        "text": {
            "Ocp-Apim-Subscription-Key": api_settings["text"]["subscription_key"],
        },
        "text_moderation": {
            "Ocp-Apim-Subscription-Key": api_settings["text_moderation"][
                "subscription_key"
            ],
            "Content-Type": "text/plain",
        },
        "translator": {
            "Ocp-Apim-Subscription-Key": api_settings["translator"]["subscription_key"],
            "Ocp-Apim-Subscription-Region": api_settings["translator"][
                "service_region"
            ],
            "Content-Type": "application/json",
        },
        "speech": {
            "Ocp-Apim-Subscription-Key": api_settings["speech"]["subscription_key"]
        },
        "spell_check": {
            "Ocp-Apim-Subscription-Key": api_settings["spell_check"][
                "subscription_key"
            ],
            "Content-Type": "application/x-www-form-urlencoded",
        },
    }


def get_microsoft_urls() -> Dict:
    api_settings = load_provider(ProviderDataEnum.KEY, "microsoft")
    return {
        "summarization": api_settings["summarization"]["url"],
        "vision": api_settings["vision"]["url"],
        "face": api_settings["face"]["url"],
        "text": api_settings["text"]["url"],
        "text_moderation": api_settings["text_moderation"]["url"],
        "translator": api_settings["translator"]["url"],
        "speech": api_settings["speech"]["url"],
        "spell_check": api_settings["spell_check"]["url"],
        "form_recognizer": api_settings["form_recognizer"]["url"],
    }


def microsoft_text_moderation_personal_infos(data):
    classification: Sequence[TextModerationItem] = []
    text_moderation: ModerationDataClass

    if classif := data.get("Classification"):
        for key, value in classif.items():
            try:
                classificator = CategoryType.choose_category_subcategory(
                    TextModerationCategoriesMicrosoftEnum[key].value
                )
                classification.append(
                    TextModerationItem(
                        label=TextModerationCategoriesMicrosoftEnum[key].value,
                        category=classificator["category"],
                        subcategory=classificator["subcategory"],
                        likelihood_score=value.get("Score", 0),
                        likelihood=standardized_confidence_score(value["Score"]),
                    )
                )
            except Exception as exc:
                continue

    text_moderation = ModerationDataClass(
        nsfw_likelihood=ModerationDataClass.calculate_nsfw_likelihood(classification),
        items=classification,
        nsfw_likelihood_score=ModerationDataClass.calculate_nsfw_likelihood_score(
            classification
        ),
    )

    return text_moderation


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
            sunglasses=None,
            reading_glasses=accessories.get("glasses", 0.0),
            swimming_goggles=None,
            face_mask=accessories.get("mask", 0.0),
            eyeglasses=None,
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
            brightness=None,
            sharpness=None,
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
            confusion=None,
            calm=None,
            unknown=None,
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
                features=FaceFeatures.default(),
            )
        )
    return deepcopy(faces_list)


def normalize_invoice_result(response):
    """normalize the original response of the provider api"""
    invoices = []
    default_dict = defaultdict(lambda: None)

    for idx in range(0, len(response.get("pages", []))):
        for document in response.get("documents", []):
            fields = document.get("fields")
            if not fields:
                continue
            customer_name = (
                fields.get("CustomerName", default_dict).get("value")
                if fields.get("CustomerName", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )
            # Customer information

            customer_id = (
                fields.get("CustomerId", default_dict).get("value")
                if fields.get("CustomerId", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            customer_tax_id = (
                fields.get("CustomerTaxId", default_dict).get("value")
                if fields.get("CustomerTaxId", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            customer_address = (
                fields.get("CustomerAddress", default_dict).get("content")
                if fields.get("CustomerAddress", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            customer_mailing_address = (
                fields.get("CustomerAddress", default_dict).get("content")
                if fields.get("CustomerAddress", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            customer_billing_address = (
                fields.get("BillingAddress", default_dict).get("content")
                if fields.get("BillingAddress", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            customer_shipping_address = (
                fields.get("ShippingAddress", default_dict).get("content")
                if fields.get("ShippingAddress", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            customer_service_address = (
                fields.get("ServiceAddress", default_dict).get("content")
                if fields.get("ServiceAddress", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            customer_remittance_address = (
                fields.get("RemittanceAddress", default_dict).get("content")
                if fields.get("RemittanceAddress", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            # Merchant information
            merchant_address = (
                fields.get("VendorAddress", default_dict).get("content")
                if fields.get("VendorAddress", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            merchant_name = (
                fields.get("VendorName", default_dict).get("value", None)
                if fields.get("VendorName", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            merchant_tax_id = (
                fields.get("VendorTaxId", default_dict).get("value")
                if fields.get("VendorTaxId", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            # Others
            purchase_order = (
                fields.get("PurchaseOrder", default_dict).get("value")
                if fields.get("PurchaseOrder", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            payment_term = (
                fields.get("PaymentTerm", default_dict).get("value")
                if fields.get("PaymentTerm", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            invoice_total = (
                (
                    fields.get("InvoiceTotal", default_dict)
                    .get("value", default_dict)
                    .get("amount")
                )
                if fields.get("InvoiceTotal", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            invoice_subtotal = (
                (
                    fields.get("SubTotal", default_dict)
                    .get("value", default_dict)
                    .get("amount")
                )
                if fields.get("SubTotal", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            invoice_number = (
                fields.get("InvoiceId", default_dict).get("value")
                if fields.get("InvoiceId", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )
            date = (
                format_date(fields.get("InvoiceDate", default_dict).get("value"))
                if fields.get("InvoiceDate", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )
            invoice_time = (
                fields.get("InvoiceTime", default_dict).get("value")
                if fields.get("InvoiceTime", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )
            date = combine_date_with_time(date, invoice_time)
            due_date = (
                format_date(fields.get("DueDate", default_dict).get("value"))
                if fields.get("DueDate", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )
            taxes = [
                TaxesInvoice(
                    value=fields.get("TotalTax", default_dict)
                    .get("value", {})
                    .get("amount")
                    if fields.get("TotalTax", {})
                    .get("bounding_regions", [{}])[0]
                    .get("page_number")
                    == idx + 1
                    else None,
                    rate=None,
                )
            ]
            amount_due = (
                fields.get("AmountDue", default_dict).get("value", {}).get("amount")
                if fields.get("AmountDue", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )
            previous_unpaid_balance = (
                fields.get("PreviousUnpaidBalance", default_dict).get("amount")
                if fields.get("PreviousUnpaidBalance", {})
                .get("bounding_regions", [{}])[0]
                .get("page_number")
                == idx + 1
                else None
            )

            # Items line
            items = fields.get("Items", default_dict).get("value", [])
            item_lines: Sequence[ItemLinesInvoice] = []
            for item in items:
                line = item.get("value", default_dict)
                if line and line is not None:
                    item_lines.append(
                        ItemLinesInvoice(
                            amount=line.get("Amount", default_dict)
                            .get("value", default_dict)
                            .get("amount")
                            if line.get("Amount", {})
                            .get("bounding_regions", [{}])[0]
                            .get("page_number")
                            == idx + 1
                            else None,
                            description=line.get("Description", default_dict).get(
                                "value"
                            )
                            if line.get("Description", {})
                            .get("bounding_regions", [{}])[0]
                            .get("page_number")
                            == idx + 1
                            else None,
                            quantity=(
                                float(
                                    (line.get("Quantity", {}) or {}).get("value", 0)
                                    or 0
                                )
                                or None
                            )
                            if line.get("Quantity", {})
                            .get("bounding_regions", [{}])[0]
                            .get("page_number")
                            == idx + 1
                            else None,
                            unit_price=line.get("UnitPrice", default_dict)
                            .get("value", default_dict)
                            .get("amount")
                            if line.get("UnitPrice", {})
                            .get("bounding_regions", [{}])[0]
                            .get("page_number")
                            else None,
                            product_code=line.get("ProductCode", default_dict).get(
                                "value"
                            )
                            if line.get("ProductCode", {})
                            .get("bounding_regions", [{}])[0]
                            .get("page_number")
                            == idx + 1
                            else None,
                            date_item=str(
                                line.get("Date", default_dict).get("value", "") or ""
                            )
                            if line.get("Date", {})
                            .get("bounding_regions", [{}])[0]
                            .get("page_number")
                            == idx + 1
                            else None,
                            tax_item=line.get("Tax", default_dict)
                            .get("value", default_dict)
                            .get("amount")
                            if fields.get("Tax", {})
                            .get("bounding_regions", [{}])[0]
                            .get("page_number")
                            == idx + 1
                            else None,
                        )
                    )

            invoices.append(
                InfosInvoiceParserDataClass(
                    customer_information=CustomerInformationInvoice(
                        customer_id=customer_id,
                        customer_name=customer_name,
                        customer_address=customer_address,
                        customer_tax_id=customer_tax_id,
                        customer_mailing_address=customer_mailing_address,
                        customer_billing_address=customer_billing_address,
                        customer_shipping_address=customer_shipping_address,
                        customer_service_address=customer_service_address,
                        customer_remittance_address=customer_remittance_address,
                        customer_email=None,
                        abn_number=None,
                        gst_number=None,
                        pan_number=None,
                        vat_number=None,
                    ),
                    merchant_information=MerchantInformationInvoice(
                        merchant_name=merchant_name,
                        merchant_address=merchant_address,
                        merchant_phone=None,
                        merchant_email=None,
                        merchant_fax=None,
                        merchant_website=None,
                        merchant_tax_id=merchant_tax_id,
                        merchant_siret=None,
                        merchant_siren=None,
                        abn_number=None,
                        gst_number=None,
                        pan_number=None,
                        vat_number=None,
                    ),
                    invoice_number=invoice_number,
                    invoice_total=invoice_total,
                    invoice_subtotal=invoice_subtotal,
                    payment_term=payment_term,
                    amount_due=amount_due,
                    previous_unpaid_balance=previous_unpaid_balance,
                    date=date,
                    due_date=due_date,
                    purchase_order=purchase_order,
                    taxes=taxes,
                    item_lines=item_lines,
                )
            )

    standardized_response = InvoiceParserDataClass(extracted_data=invoices)
    return standardized_response


def format_text_for_ssml_tags(text: str):
    tobe_replaced = [("&", "&amp"), ("<", "&lt"), (">", "&gt")]
    for element in tobe_replaced:
        text.replace(element[0], element[1])
    return text


def generate_right_ssml_text(
    text, voice_id, speaking_rate, speaking_pitch, speaking_volume
) -> str:
    attribs = {
        "rate": speaking_rate,
        "pitch": speaking_pitch,
        "volume": speaking_volume,
    }
    cleaned_attribs_string = ""
    for k, v in attribs.items():
        if not v:
            continue
        cleaned_attribs_string = f"{cleaned_attribs_string} {k}='{v}%'"

    return convert_audio_attr_in_prosody_tag(
        cleaned_attribs=cleaned_attribs_string,
        text=text,
        voice_tag=f'<voice name="{voice_id}">',
        speak_attr="version=\"1.0\" xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'",
    )


# list of audio format with there extension and a list of sample rates that it can accept
audio_format_list_extensions = [
    ("mp3", "mp3", [16000, 24000, 48000]),
    ("wav-siren", "wav", [16000]),
    ("siren", "siren", [16000]),
    ("silk", "silk", [16000, 24000]),
    ("wav", "wav", [8000, 16000, 22050, 24000, 44100, 48000]),
    ("wav-mulaw", "wav", [8000]),
    ("pcm", "pcm", [8000, 16000, 22050, 24000, 44100, 48000]),
    ("ogg", "ogg", [16000, 24000, 48000]),
    ("webm", "webm", [16000, 24000]),
    ("alaw", "alaw", [8000]),
    ("wav-alaw", "wav", [8000]),
    ("opus", "opus", [16000, 24000]),
    ("amr", "amr", [16000]),
]


def get_right_audio_support_and_sampling_rate(
    audio_format: str, sampling_rate: int, list_audio_formats: List
):
    if not audio_format:
        audio_format = "mp3"
    right_extension_sampling = next(
        filter(lambda x: x[0] == audio_format, audio_format_list_extensions), None
    )
    samplings = right_extension_sampling[2]
    if sampling_rate:
        nearest_sampling = min(samplings, key=lambda x: abs(x - sampling_rate))
    nearest_sampling = samplings[floor(len(samplings) / 2)]
    extension = right_extension_sampling[1]
    if "wav" in audio_format:
        audio_format = audio_format.replace("wav", "riff")
    if audio_format == "riff":
        audio_format = "riff-pcm"
    right_audio_format = [
        format
        for format in list_audio_formats
        if all(formt in format.lower() for formt in audio_format.split("-"))
    ]
    right_audio_format = next(
        filter(
            lambda x: f"{nearest_sampling}Hz" in x
            or f"{int(nearest_sampling/1000)}Khz" in x,
            right_audio_format,
        ),
        None,
    )
    return extension, right_audio_format


def microsoft_ocr_tables_standardize_response(
    original_response: dict,
) -> OcrTablesAsyncDataClass:
    num_pages = len(original_response["pages"])
    pages: List[Page] = [Page() for _ in range(num_pages)]

    for table in original_response.get("tables", []):
        std_table = _ocr_tables_standardize_table(table, original_response)
        page_index: int = table["boundingRegions"][0]["pageNumber"] - 1
        pages[page_index].tables.append(std_table)

    return OcrTablesAsyncDataClass(pages=pages, num_pages=num_pages)


def _ocr_tables_standardize_table(table: dict, original_response: dict) -> Table:
    num_rows = table.get("rowCount", 0)
    rows = [Row() for _ in range(num_rows)]

    for cell in table["cells"]:
        std_cell = _ocr_tables_standardize_cell(cell, original_response)
        row = rows[cell["rowIndex"]]
        row.cells.append(std_cell)

    std_table = Table(
        rows=rows, num_cols=table["columnCount"], num_rows=table["rowCount"]
    )
    return std_table


def _ocr_tables_standardize_cell(cell: dict, original_response: dict) -> Cell:
    current_page_num = cell["boundingRegions"][0]["pageNumber"]
    width = original_response["pages"][current_page_num - 1]["width"]
    height = original_response["pages"][current_page_num - 1]["height"]
    is_header = cell.get("kind") in ["columnHeader", "rowHeader"]
    bounding_box = cell["boundingRegions"][0]["polygon"]
    return Cell(
        text=cell["content"],
        col_index=cell["columnIndex"],
        row_index=cell["rowIndex"],
        row_span=cell.get("rowSpan", 1),
        col_span=cell.get("columnSpan", 1),
        is_header=is_header,
        bounding_box=BoundixBoxOCRTable(
            height=(bounding_box[7] - bounding_box[3]) / height,
            width=(bounding_box[2] - bounding_box[0]) / width,
            left=bounding_box[1] / width,
            top=bounding_box[0] / height,
        ),
        confidence=None,
    )


def _create_ocr_async_bounding_box(polygon, height, width):
    return BoundingBox(
        height=(polygon[7] - polygon[3]) / height,
        width=(polygon[2] - polygon[0]) / width,
        left=polygon[1] / width,
        top=polygon[0] / height,
    )


def _create_word(page_word, height, width):
    return Word(
        text=page_word["content"],
        bounding_box=_create_ocr_async_bounding_box(
            page_word["polygon"], height, width
        ),
        confidence=page_word["confidence"] * 100,
    )


def microsoft_ocr_async_standardize_response(
    original_response: dict,
) -> OcrAsyncDataClass:
    raw_text = original_response.get("content", "")
    pages = []

    for page in original_response.get("pages", []):
        lines = []
        height = page.get("height", 1)
        width = page.get("width", 1)
        page_words = page.get("words", [])
        i = 0
        for line in page.get("lines", []):
            text = line.get("content")
            bounding_box = _create_ocr_async_bounding_box(
                line["polygon"], height, width
            )
            words = []
            while (
                i < len(page_words)
                and page_words[i]["span"]["offset"]
                <= line["spans"][0]["offset"] + line["spans"][0]["length"]
            ):
                words.append(_create_word(page_words[i], height, width))
                i += 1
            lines.append(
                Line(text=text, words=words, bounding_box=bounding_box, confidence=None)
            )
        pages.append(OcrAsyncPage(lines=lines))
    number_of_pages = len(pages)
    return OcrAsyncDataClass(
        raw_text=raw_text, pages=pages, number_of_pages=number_of_pages
    )


def microsoft_parser_normalizer(original_response: Dict) -> List[Dict]:
    """
    Transforms the raw output from the Microsoft Invoice/Receipt Parser into a structured and user-friendly format.

    Parameters:
        original_response (dict): The original response received from the Microsoft Invoice Parser.

    Returns:
        list of dict: A formatted list containing dictionaries, each dict represent a page from the invoice.
    """
    new_response = []
    page_dict = {}
    for page_idx in range(len(original_response.get("pages") or [])):
        page_dict[page_idx] = {}

    for idx, document in enumerate(original_response.get("documents") or [{}]):
        doc_type = document.get("doc_type")
        fields = document.get("fields")
        for key_name, key_value in fields.items():
            page_index = None
            if isinstance(key_value, dict):
                page_index = (key_value.get("bounding_regions") or [{}])[0].get(
                    "page_number"
                )
            if page_index:
                page_dict[page_index - 1][key_name] = key_value
                page_dict[page_index - 1]["document_type"] = doc_type
                page_dict[page_index - 1]["document_index"] = idx + 1

            if key_name == "Items":
                items = key_value.get("value", [])
                for page_idx in range(len(original_response.get("pages") or [])):
                    page_dict[page_idx]["items"] = items

    # Convert the dictionary to a list, maintaining the order of pages
    for page_index, page_elements in sorted(page_dict.items()):
        new_response.append(page_elements)
    return new_response


def microsoft_financial_parser_formatter(
    original_response: dict,
) -> FinancialParserDataClass:
    """
    Parse a document using Microsoft financial parser (receipt/invoice) and return organized financial data.

    Args:
    - original_response (dict):  microsoft response to be parsed.

    Returns:
    - FinancialParserDataClass: Parsed financial data organized into a data class.
    """
    responses = microsoft_parser_normalizer(original_response=original_response)
    extracted_data = []
    for page_idx, page_document in enumerate(responses):
        # Customer information
        customer_information = FinancialCustomerInformation(
            name=page_document.get("CustomerName", {}).get("value"),
            id=page_document.get("CustomerId", {}).get("value"),
            tax_id=page_document.get("CustomerTaxId", {}).get("value"),
            mailling_address=page_document.get("CustomerAddress", {}).get("content"),
            billing_address=page_document.get("BillingAddress", {}).get("content"),
            shipping_address=page_document.get("ShippingAddress", {}).get("content"),
            remittance_address=page_document.get("RemittanceAddress", {}).get(
                "content"
            ),
            service_address=page_document.get("ServiceAddress", {}).get("content"),
            remit_to_name=page_document.get("CustomerAddressRecipient", {}).get(
                "content"
            ),
        )

        # Merchant information
        merchant_information = FinancialMerchantInformation(
            name=page_document.get("VendorName", {}).get("value")
            or page_document.get("MerchantName", {}).get("value"),
            address=page_document.get("VendorAddress", {}).get("content")
            or page_document.get("MerchantAddress", {}).get("content"),
            phone=page_document.get("MerchantPhoneNumber", {}).get("value"),
            tax_id=page_document.get("VendorTaxId", {}).get("value"),
            house_number=page_document.get("MerchantAddress", {})
            .get("value", {})
            .get("house_number"),
            street_name=page_document.get("MerchantAddress", {})
            .get("value", {})
            .get("street_address"),
            city=page_document.get("MerchantAddress", {})
            .get("value", {})
            .get("city_district"),
            zip_code=page_document.get("MerchantAddress", {})
            .get("value", {})
            .get("postal_code"),
            province=page_document.get("MerchantAddress", {})
            .get("value", {})
            .get("state_district"),
        )

        # Payment information
        payment_information = FinancialPaymentInformation(
            total=page_document.get("InvoiceTotal", {}).get("value", {}).get("amount"),
            subtotal=page_document.get("SubTotal", {}).get("value", {}).get("amount"),
            payment_terms=page_document.get("PaymentTerm", {}).get("value"),
            amount_due=page_document.get("AmountDue", {})
            .get("value", {})
            .get("amount"),
            previous_unpaid_balance=page_document.get("PreviousUnpaidBalance", {}).get(
                "amount"
            ),
            total_tax=page_document.get("TotalTax", {}).get("value", {}).get("amount")
            if isinstance(page_document.get("TotalTax", {}).get("value"), dict)
            else page_document.get("TotalTax", {}).get("value"),
            discount=page_document.get("TotalDiscount", {})
            .get("value", {})
            .get("amount"),
        )

        # Document information
        financial_document_information = FinancialDocumentInformation(
            invoice_receipt_id=page_document.get("InvoiceId", {}).get("value"),
            purchase_order=page_document.get("PurchaseOrder", {}).get("value"),
            invoice_due_date=format_date(page_document.get("DueDate", {}).get("value")),
            invoice_date=format_date(
                page_document.get("InvoiceDate", {}).get("value")
                or page_document.get("TransactionDate", {}).get("value")
            ),
            time=convert_time_to_string(
                page_document.get("InvoiceTime", {}).get("value")
                or page_document.get("TransactionTime", {}).get("value")
            ),
        )

        # Bank information
        payment_details = page_document.get("PaymentDetails", {}).get("value", [])
        payment_items = []
        for obj in payment_details:
            line = obj.get("value", {})
            if line:
                payment_items.append(
                    {
                        "iban": line.get("IBAN", {}).get("content"),
                        "swift": line.get("SWIFT", {}).get("content"),
                    }
                )
        bank = FinancialBankInformation(
            swift=payment_items[0].get("swift") if len(payment_items) > 0 else None,
            iban=payment_items[0].get("iban") if len(payment_items) > 0 else None,
        )

        # Local information
        local = FinancialLocalInformation(
            currency_code=page_document.get("InvoiceTotal", {})
            .get("value", {})
            .get("code")
        )

        # Document metadata
        document_metadata = FinancialDocumentMetadata(
            document_index=page_document.get("document_index"),
            document_page_number=page_idx + 1,
            document_type=page_document.get("document_type"),
        )

        # Items
        items = page_document.get("items")
        item_lines = []
        if items:
            for item in items:
                page_item = item["bounding_regions"][0].get("page_number")
                line = item.get("value", {})
                if page_item == page_idx + 1 and line:
                    item_lines.append(
                        FinancialLineItem(
                            amount_line=line.get("Amount", {})
                            .get("value", {})
                            .get("amount")
                            or line.get("TotalPrice", {}).get("value"),
                            description=line.get("Description", {}).get("value"),
                            quantity=line.get("Quantity", {}).get("value") or 0,
                            unit_price=line.get("UnitPrice", {})
                            .get("value", {})
                            .get("amount"),
                            product_code=line.get("ProductCode", {}).get("value"),
                            date=line.get("Date", {}).get("value").isoformat()
                            if isinstance(
                                line.get("Date", {}).get("value"), datetime.date
                            )
                            else line.get("Date", {}).get("value"),
                            tax=(line.get("Tax", {}).get("value") or {}).get("amount"),
                            tax_rate=line.get("TaxRate", {}).get("value"),
                        )
                    )
        extracted_data.append(
            FinancialParserObjectDataClass(
                customer_information=customer_information,
                merchant_information=merchant_information,
                payment_information=payment_information,
                financial_document_information=financial_document_information,
                bank=bank,
                local=local,
                document_metadata=document_metadata,
                item_lines=item_lines,
            )
        )
    return FinancialParserDataClass(extracted_data=extracted_data)
