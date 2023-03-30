from collections import defaultdict
from copy import deepcopy
import json
from typing import Dict, Optional, Sequence
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
from edenai_apis.features.text import (
    ModerationDataClass,
    TextModerationItem,
    TextModerationCategoriesMicrosoftEnum
)
from edenai_apis.features.ocr.identity_parser.identity_parser_dataclass import format_date
from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider

from edenai_apis.utils.conversion import combine_date_with_time, standardized_confidence_score


def get_microsoft_headers() -> Dict:
    api_settings = load_provider(ProviderDataEnum.KEY, "microsoft")
    return {
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
            "text_moderation": {
                "Ocp-Apim-Subscription-Key": api_settings["text_moderation"][
                    "subscription_key"
                ],
                "Content-Type": "text/plain"
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
            "spell_check": {
                "Ocp-Apim-Subscription-Key": api_settings["spell_check"]["subscription_key"],
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
            "text_moderation" : api_settings["text_moderation"]["url"],
            "translator": api_settings["translator"]["url"],
            "speech": api_settings["speech"]["url"],
            "spell_check": api_settings["spell_check"]["url"],
        }

def microsoft_text_moderation_personal_infos(data):
    classification : Sequence[TextModerationItem] = []
    text_moderation : ModerationDataClass

    if classif := data.get("Classification"):
        for key, value in classif.items():
            try:
                classification.append(
                    TextModerationItem(
                        label= TextModerationCategoriesMicrosoftEnum[key].value,
                        likelihood= standardized_confidence_score(value["Score"])
                    )
                )
            except Exception as exc:
                continue

    text_moderation = ModerationDataClass(
        nsfw_likelihood= ModerationDataClass.calculate_nsfw_likelihood(classification),
        items= classification
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

def normalize_invoice_result(response):
    """normalize the original response of the provider api"""
    invoices = []
    default_dict = defaultdict(lambda: None)
    for document in response.get("documents", []):
        fields = document.get("fields")
        if not fields:
            continue
        customer_name = fields.get("CustomerName", default_dict).get("value")
        # Customer information

        customer_id = fields.get("CustomerId", default_dict).get("value")
        customer_tax_id = fields.get("CustomerTaxId", default_dict).get("value")
        customer_address = fields.get("CustomerAddress", default_dict).get("content")
        customer_mailing_address = fields.get("CustomerAddress", default_dict).get(
            "content"
        )
        customer_billing_address = fields.get("BillingAddress", default_dict).get("content")
        customer_shipping_address = fields.get("ShippingAddress", default_dict).get(
            "content"
        )
        customer_service_address = fields.get("ServiceAddress", default_dict).get("content")
        customer_remittance_address = fields.get("RemittanceAddress", default_dict).get(
            "content"
        )

        # Merchant information
        merchant_address = fields.get("VendorAddress", default_dict).get("content")
        merchant_name = fields.get("VendorName", default_dict).get("value", None)
        merchant_tax_id = fields.get("VendorTaxId", default_dict).get("value")

        # Others
        purchase_order = fields.get("PurchaseOrder", default_dict).get("value")
        payment_term = fields.get("PaymentTerm", default_dict).get("value")
        invoice_total = fields.get("InvoiceTotal", default_dict).get("value", default_dict).get("amount")
        invoice_subtotal = fields.get("SubTotal", default_dict).get("value", default_dict).get("amount")
        invoice_number = (
            fields.get("InvoiceId", default_dict)
            .get("value")
        )
        date = format_date(fields.get("InvoiceDate", default_dict).get("value"))
        invoice_time = fields.get("InvoiceTime", default_dict).get("value")
        date = combine_date_with_time(date, invoice_time)
        due_date = format_date(fields.get("DueDate", default_dict).get("value"))
        taxes = [
            TaxesInvoice(value=fields.get("TotalTax", default_dict).get("value", {}).get("amount"))
        ]
        amount_due = fields.get("AmountDue", default_dict).get("value", {}).get("amount")
        service_start_date = fields.get("ServiceStartDate", default_dict).get("value")
        service_end_date = fields.get("ServiceEndDate", default_dict).get("value")
        previous_unpaid_balance = fields.get("PreviousUnpaidBalance", default_dict).get(
            "amount"
        )

        # Items line
        items = fields.get("Items", default_dict).get("value", [])
        item_lines: Sequence[ItemLinesInvoice] = []
        for item in items:
            line = item.get("value", default_dict)
            if line:
                item_lines.append(
                    ItemLinesInvoice(
                        amount=line.get("Amount", default_dict).get("value", default_dict).get("amount"),
                        description=line.get("Description", default_dict).get("value"),
                        quantity=line.get("Quantity", default_dict).get("value"),
                        unit_price=line.get("UnitPrice", default_dict).get("value", default_dict).get("amount"),
                        product_code=line.get("ProductCode", default_dict).get("value"),
                        date_item=line.get("Date", default_dict).get("value"),
                        tax_item=line.get("Tax", default_dict).get("value", default_dict).get("amount"),
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
                ),
                merchant_information=MerchantInformationInvoice(
                    merchant_name=merchant_name,
                    merchant_address=merchant_address,
                    merchant_tax_id=merchant_tax_id,
                ),
                invoice_number=invoice_number,
                invoice_total=invoice_total,
                invoice_subtotal=invoice_subtotal,
                payment_term=payment_term,
                amount_due=amount_due,
                service_start_date=service_start_date,
                service_end_date=service_end_date,
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
