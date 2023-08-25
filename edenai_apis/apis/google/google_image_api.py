from io import BufferedReader
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
from time import time, sleep
import uuid
import numpy as np
import requests
from edenai_apis.apis.google.google_helpers import (
    get_access_token,
    get_long_operation_status,
    handle_google_call,
    score_to_content,
)
from edenai_apis.features.image.automl_classification import (
    AutomlClassificationCreateDataset,
    AutomlClassificationTraining,
    TrainingModelMetrics,
    AutomlClassificationCreateEndpoint,
    AutomlClassificationDeployModel,
    AutomlClassificationPrediction,
)
from edenai_apis.features.image.explicit_content.explicit_content_dataclass import (
    ExplicitContentDataClass,
    ExplicitItem,
)
from edenai_apis.features.image.face_detection.face_detection_dataclass import (
    FaceAccessories,
    FaceBoundingBox,
    FaceDetectionDataClass,
    FaceEmotions,
    FaceFacialHair,
    FaceFeatures,
    FaceHair,
    FaceItem,
    FaceLandmarks,
    FaceMakeup,
    FaceOcclusions,
    FacePoses,
    FaceQuality,
)
from edenai_apis.features.image.image_interface import ImageInterface
from edenai_apis.features.image.landmark_detection.landmark_detection_dataclass import (
    LandmarkDetectionDataClass,
    LandmarkItem,
    LandmarkLatLng,
    LandmarkLocation,
    LandmarkVertice,
)
from edenai_apis.features.image.logo_detection.logo_detection_dataclass import (
    LogoBoundingPoly,
    LogoDetectionDataClass,
    LogoItem,
    LogoVertice,
)
from edenai_apis.features.image.object_detection.object_detection_dataclass import (
    ObjectDetectionDataClass,
    ObjectItem,
)
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.types import (
    AsyncBaseResponseType,
    AsyncLaunchJobResponseType,
    AsyncPendingResponseType,
    AsyncResponseType,
    ResponseType,
)
from PIL import Image as Img

from google.cloud import vision, storage
from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse
from google.protobuf.json_format import MessageToDict


class GoogleImageApi(ImageInterface):
    def image__explicit_content(
        self, file: str, file_url: str = ""
    ) -> ResponseType[ExplicitContentDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)

        payload = {"image": image}
        response = handle_google_call(
            self.clients["image"].safe_search_detection, **payload
        )

        # Convert response to dict
        data = AnnotateImageResponse.to_dict(response)

        if data.get("error") is not None:
            raise ProviderException(data["error"])

        original_response = data.get("safe_search_annotation", {})

        items = []
        for safe_search_annotation, likelihood in original_response.items():
            items.append(
                ExplicitItem(
                    label=safe_search_annotation.capitalize(), likelihood=likelihood
                )
            )

        nsfw_likelihood = ExplicitContentDataClass.calculate_nsfw_likelihood(items)

        return ResponseType(
            original_response=original_response,
            standardized_response=ExplicitContentDataClass(
                items=items, nsfw_likelihood=nsfw_likelihood
            ),
        )

    def image__object_detection(
        self, file: str, model: str = None, file_url: str = ""
    ) -> ResponseType[ObjectDetectionDataClass]:
        file_ = open(file, "rb")
        image = vision.Image(content=file_.read())

        payload = {"image": image}
        response = handle_google_call(
            self.clients["image"].object_localization, **payload
        )
        response = MessageToDict(response._pb)

        file_.close()
        items = []
        for object_annotation in response.get("localizedObjectAnnotations", []):
            x_min, x_max = np.infty, -np.infty
            y_min, y_max = np.infty, -np.infty
            # Getting borders
            for normalize_vertice in object_annotation["boundingPoly"][
                "normalizedVertices"
            ]:
                x_min, x_max = min(x_min, normalize_vertice.get("x", 0)), max(
                    x_max, normalize_vertice.get("x", 0)
                )
                y_min, y_max = min(y_min, normalize_vertice.get("y", 0)), max(
                    y_max, normalize_vertice.get("y", 0)
                )
                items.append(
                    ObjectItem(
                        label=object_annotation["name"],
                        confidence=object_annotation["score"],
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                    )
                )

        return ResponseType[ObjectDetectionDataClass](
            original_response=response,
            standardized_response=ObjectDetectionDataClass(items=items),
        )

    def image__face_detection(
        self, file: str, file_url: str = ""
    ) -> ResponseType[FaceDetectionDataClass]:
        with open(file, "rb") as file_:
            file_content = file_.read()
        img_size = Img.open(file).size
        image = vision.Image(content=file_content)

        payload = {"image": image, "max_results": 100}
        response = handle_google_call(self.clients["image"].face_detection, **payload)
        original_result = MessageToDict(response._pb)

        result = []
        width, height = img_size
        for face in original_result.get("faceAnnotations", []):
            # emotions
            emotions = FaceEmotions(
                joy=score_to_content(face.get("joyLikelihood")),
                sorrow=score_to_content(face.get("sorrowLikelihood")),
                anger=score_to_content(face.get("angerLikelihood")),
                surprise=score_to_content(face.get("surpriseLikelihood")),
                # Not supported by Google
                # ------------------------
                disgust=None,
                fear=None,
                confusion=None,
                calm=None,
                contempt=None,
                unknown=None,
                neutral=None,
                # ------------------------
            )

            # quality
            quality = FaceQuality(
                exposure=2
                * score_to_content(face.get("underExposedLikelihood", 0))
                / 10,
                blur=2 * score_to_content(face.get("blurredLikelihood", 0)) / 10,
                noise=None,
                brightness=None,
                sharpness=None,
            )

            # accessories
            accessories = FaceAccessories.default()
            accessories.headwear = (
                2 * score_to_content(face.get("headwearLikelihood", 0)) / 10
            )

            # landmarks
            landmark_output = {}
            for land in face.get("landmarks", []):
                if "type" in land and "UNKNOWN_LANDMARK" not in land:
                    landmark_output[land["type"]] = [
                        land["position"]["x"] / width,
                        land["position"]["y"] / height,
                    ]
            landmarks = FaceLandmarks(
                left_eye=landmark_output.get("LEFT_EYE", []),
                left_eye_top=landmark_output.get("LEFT_EYE_TOP_BOUNDARY", []),
                left_eye_right=landmark_output.get("LEFT_EYE_RIGHT_CORNER", []),
                left_eye_bottom=landmark_output.get("LEFT_EYE_BOTTOM_BOUNDARY", []),
                left_eye_left=landmark_output.get("LEFT_EYE_LEFT_CORNER", []),
                right_eye=landmark_output.get("RIGHT_EYE", []),
                right_eye_top=landmark_output.get("RIGHT_EYE_TOP_BOUNDARY", []),
                right_eye_right=landmark_output.get("RIGHT_EYE_RIGHT_CORNER", []),
                right_eye_bottom=landmark_output.get("RIGHT_EYE_BOTTOM_BOUNDARY", []),
                right_eye_left=landmark_output.get("RIGHT_EYE_LEFT_CORNER", []),
                left_eyebrow_left=landmark_output.get("LEFT_OF_LEFT_EYEBROW", []),
                left_eyebrow_right=landmark_output.get("LEFT_OF_RIGHT_EYEBROW", []),
                left_eyebrow_top=landmark_output.get("LEFT_EYEBROW_UPPER_MIDPOINT", []),
                right_eyebrow_left=landmark_output.get("RIGHT_OF_LEFT_EYEBROW", []),
                right_eyebrow_right=landmark_output.get("RIGHT_OF_RIGHT_EYEBROW", []),
                nose_tip=landmark_output.get("NOSE_TIP", []),
                nose_bottom_right=landmark_output.get("NOSE_BOTTOM_RIGHT", []),
                nose_bottom_left=landmark_output.get("NOSE_BOTTOM_LEFT", []),
                mouth_left=landmark_output.get("MOUTH_LEFT", []),
                mouth_right=landmark_output.get("MOUTH_RIGHT", []),
                right_eyebrow_top=landmark_output.get(
                    "RIGHT_EYEBROW_UPPER_MIDPOINT", []
                ),
                midpoint_between_eyes=landmark_output.get("MIDPOINT_BETWEEN_EYES", []),
                nose_bottom_center=landmark_output.get("NOSE_BOTTOM_CENTER", []),
                upper_lip=landmark_output.get("GET_UPPER_LIP", []),
                under_lip=landmark_output.get("GET_LOWER_LIP", []),
                mouth_center=landmark_output.get("MOUTH_CENTER", []),
                left_ear_tragion=landmark_output.get("LEFT_EAR_TRAGION", []),
                right_ear_tragion=landmark_output.get("RIGHT_EAR_TRAGION", []),
                forehead_glabella=landmark_output.get("FOREHEAD_GLABELLA", []),
                chin_gnathion=landmark_output.get("CHIN_GNATHION", []),
                chin_left_gonion=landmark_output.get("CHIN_LEFT_GONION", []),
                chin_right_gonion=landmark_output.get("CHIN_RIGHT_GONION", []),
                left_cheek_center=landmark_output.get("LEFT_CHEEK_CENTER", []),
                right_cheek_center=landmark_output.get("RIGHT_CHEEK_CENTER", []),
            )

            # bounding box
            bounding_poly = face.get("fdBoundingPoly", {}).get("vertices", [])

            result.append(
                FaceItem(
                    accessories=accessories,
                    quality=quality,
                    emotions=emotions,
                    landmarks=landmarks,
                    poses=FacePoses(
                        roll=face.get("rollAngle"),
                        pitch=face.get("panAngle"),
                        yaw=face.get("tiltAngle"),
                    ),
                    confidence=face.get("detectionConfidence"),
                    # indices are this way because array of bounding boxes
                    # follow this pattern:
                    # [top-left, top-right, bottom-right, bottom-left]
                    bounding_box=FaceBoundingBox(
                        x_min=bounding_poly[0].get("x", 0.0) / width,
                        x_max=bounding_poly[1].get("x", width) / width,
                        y_min=bounding_poly[0].get("y", 0.0) / height,
                        y_max=bounding_poly[3].get("y", height) / height,
                    ),
                    # Not supported by Google Cloud Vision
                    # --------------------
                    age=None,
                    gender=None,
                    hair=FaceHair.default(),
                    facial_hair=FaceFacialHair.default(),
                    makeup=FaceMakeup.default(),
                    occlusions=FaceOcclusions.default(),
                    features=FaceFeatures.default(),
                    # --------------------
                )
            )
        return ResponseType[FaceDetectionDataClass](
            original_response=original_result,
            standardized_response=FaceDetectionDataClass(items=result),
        )

    def image__landmark_detection(
        self, file: str, file_url: str = ""
    ) -> ResponseType[LandmarkDetectionDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)
        payload = {"image": image}
        response = handle_google_call(
            self.clients["image"].landmark_detection, **payload
        )
        dict_response = vision.AnnotateImageResponse.to_dict(response)
        landmarks = dict_response.get("landmark_annotations", [])

        items: Sequence[LandmarkItem] = []
        for landmark in landmarks:
            if landmark.get("description") not in [item.description for item in items]:
                vertices: Sequence[LandmarkVertice] = []
                for poly in landmark.get("bounding_poly", {}).get("vertices", []):
                    vertices.append(LandmarkVertice(x=poly["x"], y=poly["y"]))
                locations = []
                for location in landmark.get("locations", []):
                    locations.append(
                        LandmarkLocation(
                            lat_lng=LandmarkLatLng(
                                latitude=location.get("lat_lng", {}).get("latitude"),
                                longitude=location.get("lat_lng", {}).get("longitude"),
                            )
                        )
                    )
                items.append(
                    LandmarkItem(
                        description=landmark.get("description"),
                        confidence=landmark.get("score"),
                        bounding_box=vertices,
                        locations=locations,
                    )
                )
        if dict_response.get("error"):
            raise ProviderException(
                message=dict_response["error"].get(
                    "message", "Error calling Google Api"
                )
            )

        return ResponseType[LandmarkDetectionDataClass](
            original_response=landmarks,
            standardized_response=LandmarkDetectionDataClass(items=items),
        )

    def image__logo_detection(
        self, file: str, file_url: str = ""
    ) -> ResponseType[LogoDetectionDataClass]:
        with open(file, "rb") as file_:
            content = file_.read()
        image = vision.Image(content=content)

        payload = {"image": image}
        response = handle_google_call(self.clients["image"].logo_detection, **payload)

        response = MessageToDict(response._pb)

        float_or_none = lambda val: float(val) if val else None
        # Handle error
        if response.get("error", {}).get("message") is not None:
            raise ProviderException(response["error"]["message"])

        items: Sequence[LogoItem] = []
        for key in response.get("logoAnnotations", []):
            vertices = []
            for vertice in key.get("boundingPoly").get("vertices"):
                vertices.append(
                    LogoVertice(
                        x=float_or_none(vertice.get("x")),
                        y=float_or_none(vertice.get("y")),
                    )
                )

            items.append(
                LogoItem(
                    description=key.get("description"),
                    score=key.get("score"),
                    bounding_poly=LogoBoundingPoly(vertices=vertices),
                )
            )
        return ResponseType[LogoDetectionDataClass](
            original_response=response,
            standardized_response=LogoDetectionDataClass(items=items),
        )

    def image__automl_classification__create_dataset_async__launch_job(
        self,
        file: str,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:
        url_subdomain = "us-central1-aiplatform"
        project_id = self.project_id
        token = get_access_token(self.location)
        bucket_region = self.api_settings["bucket_region"]
        bucket_name = "automl-vision-classification"
        storage_client: storage.Client = self.clients["storage"]
        bucket = storage_client.get_bucket(bucket_name)

        url = f"https://{url_subdomain}.googleapis.com/v1/projects/{project_id}/locations/{bucket_region}/datasets"

        file_name = Path(file).stem
        display_name = f"{file_name}_{str(int(time()))}"

        # upload csv file into a bucket
        destination_blob_name = f"training/{display_name}.csv"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file)

        payload = {
            "display_name": display_name,
            "metadata_schema_uri": "gs://google-cloud-aiplatform/schema/dataset/metadata/image_1.0.0.yaml",
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        # create a dataset
        request = requests.post(url=url, headers=headers, json=payload)
        if request.status_code > 201:
            raise ProviderException(
                "Something went wrong when creating the dataset",
                code=requests.status_codes,
            )
        response = request.json()
        operation_name = response["name"]
        print("The dataset is being created...")
        operation_status = False
        while not operation_status:
            sleep(0.5)
            response = get_long_operation_status(
                bucket_region, operation_name, self.location, token
            )
            operation_status = response.get("done") or False
        print("Dataset created...")

        # import data into the dataset
        dataset_id = operation_name.split("/")[5]

        # uris
        input_uris = f"gs://{bucket_name}/{destination_blob_name}"
        url = f"https://{url_subdomain}.googleapis.com/v1/projects/{self.project_id}/locations/{bucket_region}/datasets/{dataset_id}:import"
        payload = {
            "import_configs": [
                {
                    "gcs_source": {"uris": input_uris},
                    "import_schema_uri": "gs://google-cloud-aiplatform/schema/dataset/ioformat/image_classification_single_label_io_format_1.0.0.yaml",
                }
            ]
        }

        request = requests.post(url=url, headers=headers, json=payload)
        if request.status_code > 201:
            raise ProviderException(
                "Something went wrong when creating the dataset",
                code=requests.status_codes,
            )
        response = request.json()
        operation_name = response["name"]
        dataset_job_id = f"{operation_name.split('/')[-1]}EdenAI{dataset_id}"

        return AsyncLaunchJobResponseType(provider_job_id=dataset_job_id)

    def image__automl_classification__create_dataset_async__get_job_result(
        self, dataset_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationCreateDataset]:
        opreration_id, dataset_id = dataset_job_id.split("EdenAI")
        location = self.api_settings["bucket_region"]
        project_id = self.project_id
        token = get_access_token(self.location)
        operation_name = f"projects/{project_id}/locations/{location}/datasets/{dataset_id}/operations/{opreration_id}"

        response = get_long_operation_status(
            location, operation_name, self.location, token
        )

        operation_status = response.get("done") or False
        operation_response = response.get("response")

        if not operation_status:
            return AsyncPendingResponseType[AutomlClassificationCreateDataset](
                provider_job_id=dataset_job_id
            )
        if operation_status and not operation_response:
            raise ProviderException(response.get("error"))

        standardized_response = AutomlClassificationCreateDataset(dataset_id=dataset_id)
        return AsyncResponseType[AutomlClassificationCreateDataset](
            original_response=dataset_id,
            standardized_response=standardized_response,
            provider_job_id=dataset_job_id,
        )

    def image__automl_classification__training_async__launch_job(
        self,
        dataset_id: str,
        model_name: str = "",
        model_description: str = "",
        fraction_split: Tuple[float] = (),
    ) -> AsyncLaunchJobResponseType:
        url_subdomain = "us-central1-aiplatform"
        project_id = self.project_id
        token = get_access_token(self.location)
        bucket_region = self.api_settings["bucket_region"]

        url = f"https://{url_subdomain}.googleapis.com/v1/projects/{project_id}/locations/{bucket_region}/trainingPipelines"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        payload_fraction_split = {}
        if fraction_split:
            if len(fraction_split) != 3:
                raise ProviderException(
                    "Should provide all of training, validation and test fractions"
                )
            train, valid, test = fraction_split
            if train + valid + test != 1:
                raise ProviderException("Some of all fractions should be equal to 1.0")
            payload_fraction_split = {
                "trainingFraction": train,
                "validationFraction": valid,
                "testFraction": test,
            }

        payload = {
            "displayName": str(uuid.uuid4()),
            "inputDataConfig": {
                "datasetId": dataset_id,
                "fractionSplit": payload_fraction_split,
            },
            "modelToUpload": {
                "displayName": model_name,
                "description": model_description,
            },
            "trainingTaskDefinition": "gs://google-cloud-aiplatform/schema/trainingjob/definition/automl_image_classification_1.0.0.yaml",
            "trainingTaskInputs": {
                "multiLabel": "false",
                "modelType": ["CLOUD"],
                "budgetMilliNodeHours": 400000,
            },
        }

        request = requests.post(url=url, headers=headers, json=payload)
        if request.status_code > 201:
            raise ProviderException(
                "Something went wrong when creating the dataset",
                code=requests.status_codes,
            )
        response = request.json()
        operation_name = response["name"]

        return AsyncLaunchJobResponseType(provider_job_id=operation_name.split("/")[-1])

    def image__automl_classification__training_async__get_job_result(
        self, training_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationTraining]:
        url_subdomain = "us-central1-aiplatform"
        pipeline_training_id = training_job_id
        location = self.api_settings["bucket_region"]
        project_id = self.project_id
        bucket_region = self.api_settings["bucket_region"]
        token = get_access_token(self.location)

        operation_name = f"projects/{project_id}/locations/{location}/trainingPipelines/{pipeline_training_id}"

        response = get_long_operation_status(
            location, operation_name, self.location, token
        )

        training_status = response.get("state")
        if training_status == "PIPELINE_STATE_RUNNING":
            return AsyncPendingResponseType(provider_job_id=training_job_id)
        if training_status != "PIPELINE_STATE_SUCCEEDED":
            error_msg = response.get("error") or "training failed!!"
            raise ProviderException(error_msg)

        # get model id
        model_to_upload = response.get("modelToUpload", {}) or {}
        model_complete_name = model_to_upload.get("name", "") or ""
        try:
            model_id = model_complete_name.split("/")[-1]
        except:
            raise ProviderException("Model was not created !!")

        standardized_response = AutomlClassificationTraining(model_id=model_id)
        # get metrics
        url = f"https://{url_subdomain}.googleapis.com/v1/projects/{project_id}/locations/{bucket_region}/models/{model_id}/evaluations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        request = requests.get(url, headers=headers)
        response = request.json()
        if request.status_code >= 400:
            pass  # should remove metrics from response
        model_evaluation = response.get("modelEvaluations", []) or []
        if len(model_evaluation) == 0:  # no metrics
            return AsyncResponseType[AutomlClassificationTraining](
                original_response=model_id,
                standardized_response=standardized_response,
                provider_job_id=training_job_id,
            )
        model_evaluation = model_evaluation[0]
        model_evaluation_name = model_evaluation.get("name", "") or ""
        try:
            evaluation_id = model_evaluation_name.split("/")[-1]
        except:  # the is no evaluation, should remove metrics from response
            return AsyncResponseType[AutomlClassificationTraining](
                original_response=model_id,
                standardized_response=standardized_response,
                provider_job_id=training_job_id,
            )
        metrics = model_evaluation.get("metrics", {}) or {}
        au_prc = metrics.get("auPrc")
        log_loss = metrics.get("logLoss")
        recall = metrics.get("confidenceMetrics")[1]["recall"]
        precision = metrics.get("confidenceMetrics")[1]["precision"]

        metric_instance = TrainingModelMetrics(
            au_prc=au_prc, log_loss=log_loss, recall=recall, precision=precision
        )
        standardized_response.metrics = metric_instance
        return AsyncResponseType[AutomlClassificationTraining](
            original_response={"model_id": model_id, "metrics": metrics},
            standardized_response=standardized_response,
            provider_job_id=training_job_id,
        )

    def image__automl_classification__create_endpoint_async__launch_job(
        self, model_id: str
    ) -> AsyncLaunchJobResponseType:
        url_subdomain = "us-central1-aiplatform"
        location = self.api_settings["bucket_region"]
        project_id = self.project_id
        token = get_access_token(self.location)

        # create an endpoint for the deployement
        endpoint_name = f"{model_id}-{str(int(time()))}"
        url = f"https://{url_subdomain}.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        payload = {"display_name": endpoint_name}
        request = requests.post(url, headers=headers, json=payload)
        response = request.json()
        if request.status_code >= 400:
            raise ProviderException("Can't deploy model !!", code=request.status_code)
        operation_name = response["name"]
        endpoint_id = operation_name.split("/")[5]

        return AsyncLaunchJobResponseType(
            provider_job_id=f"{operation_name.split('/')[-1]}EdenAI{endpoint_id}"
        )

    def image__automl_classification__create_endpoint_async__get_job_result(
        self, endpoint_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationCreateEndpoint]:
        operation_id, endpoint_id = endpoint_job_id.split("EdenAI")
        location = self.api_settings["bucket_region"]
        project_id = self.project_id
        token = get_access_token(self.location)

        operation_name = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}/operations/{operation_id}"

        response = get_long_operation_status(
            location, operation_name, self.location, token
        )

        operation_status = response.get("done") or False
        operation_response = response.get("response")

        if not operation_status:
            return AsyncPendingResponseType(provider_job_id=endpoint_job_id)
        if operation_status and not operation_response:
            raise ProviderException(response.get("error"))

        return AsyncResponseType[AutomlClassificationCreateEndpoint](
            original_response=endpoint_id,
            standardized_response=AutomlClassificationCreateEndpoint(
                endpoint_id=endpoint_id
            ),
            provider_job_id=endpoint_job_id,
        )

    def image__automl_classification__deploy_model_async__launch_job(
        self, model_id: str, endpoint_id: str, deploy_model_name: str
    ) -> AsyncLaunchJobResponseType:
        url_subdomain = "us-central1-aiplatform"
        location = self.api_settings["bucket_region"]
        project_id = self.project_id
        bucket_region = self.api_settings["bucket_region"]
        token = get_access_token(self.location)

        # deploy the model
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        url = f"https://{url_subdomain}.googleapis.com/v1/projects/{project_id}/locations/{bucket_region}/endpoints/{endpoint_id}:deployModel"
        payload = {
            "deployedModel": {
                "model": f"projects/{project_id}/locations/{bucket_region}/models/{model_id}",
                "displayName": deploy_model_name,
                "automaticResources": {"minReplicaCount": 1, "maxReplicaCount": 1},
            }
        }
        request = requests.post(url, headers=headers, json=payload)
        response = request.json()
        if request.status_code >= 400:
            error = (response.get("error", {}) or {}).get(
                "message", ""
            ) or "Could not deploy model !!"
            raise ProviderException(error, code=request.status_code)
        operation_name = response["name"]

        return AsyncLaunchJobResponseType(
            provider_job_id=f"{operation_name.split('/')[-1]}EdenAI{endpoint_id}"
        )

    def image__automl_classification__deploy_model_async__get_job_result(
        self, deployement_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationDeployModel]:
        operation_id, endpoint_id = deployement_job_id.split("EdenAI")
        location = self.api_settings["bucket_region"]
        project_id = self.project_id
        token = get_access_token(self.location)

        operation_name = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}/operations/{operation_id}"

        response = get_long_operation_status(
            location, operation_name, self.location, token
        )

        operation_status = response.get("done") or False
        operation_response = response.get("response")

        if not operation_status:
            return AsyncPendingResponseType(provider_job_id=deployement_job_id)
        if operation_status and not operation_response:
            raise ProviderException(response.get("error"))

        return AsyncResponseType[AutomlClassificationDeployModel](
            original_response=True,
            standardized_response=AutomlClassificationDeployModel(deployed=True),
            provider_job_id=deployement_job_id,
        )

    def image__automl_classification__prediction_async__launch_job(
        self,
        file: str,
        mime_type: str,
        model_id: str,
        job_name: str,
        file_url: str = "",
    ) -> AsyncLaunchJobResponseType:
        bucket_result = "automl-vision-classification-predictions"
        url_subdomain = "us-central1-aiplatform"
        project_id = self.project_id
        token = get_access_token(self.location)
        bucket_region = self.api_settings["bucket_region"]
        bucket_name = "automl-vision-classification"
        storage_client: storage.Client = self.clients["storage"]
        bucket = storage_client.get_bucket(bucket_name)

        # upload file to bucket
        _, ext = os.path.splitext(file)
        file_name = Path(file).stem
        display_name = f"{file_name}_{str(int(time()))}"

        destination_blob_name = f"prediction/{display_name}{ext}"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file)

        # create a json file pointing to the file ressource
        ressource = {
            "content": f"gs://{bucket_name}/{destination_blob_name}",
            "mimeType": mime_type,
        }
        json_destination_blob_name = f"prediction/{display_name}.json"
        blob = bucket.blob(json_destination_blob_name)
        blob.upload_from_string(json.dumps(ressource))

        payload = {
            "displayName": job_name,
            "model": f"projects/{project_id}/locations/{bucket_region}/models/{model_id}",
            "modelParameters": {},
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {
                    "uris": [f"gs://{bucket_name}/{json_destination_blob_name}"]
                },
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": f"gs://{bucket_result}"},
            },
        }
        url = f"https://{url_subdomain}.googleapis.com/v1/projects/{project_id}/locations/{bucket_region}/batchPredictionJobs"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        request = requests.post(url, headers=headers, json=payload)
        response = request.json()
        if request.status_code >= 400:
            error = (
                response.get("error")
                or "Something went wrong when doing the prediction !!"
            )
            raise ProviderException(error, code=request.status_code)
        operation_name = response["name"]
        prediction_job_id = operation_name.split("/")[-1]

        return AsyncLaunchJobResponseType(provider_job_id=prediction_job_id)

    def image__automl_classification__prediction_async__get_job_result(
        self, prediction_job_id: str
    ) -> AsyncBaseResponseType[AutomlClassificationPrediction]:
        location = self.api_settings["bucket_region"]
        project_id = self.project_id
        token = get_access_token(self.location)
        storage_client: storage.Client = self.clients["storage"]

        operation_name = f"projects/{project_id}/locations/{location}/batchPredictionJobs/{prediction_job_id}"

        response = get_long_operation_status(
            location, operation_name, self.location, token
        )
        job_status = response.get("state", "")
        if job_status == "JOB_STATE_PENDING" or job_status == "JOB_STATE_RUNNING":
            return AsyncPendingResponseType(provider_job_id=prediction_job_id)
        if job_status != "JOB_STATE_SUCCEEDED":
            error = (
                response.get("error", "")
                or "Something went wrong when getting prediction results!!"
            )
            raise ProviderException(error)
        output_dir = response.get("outputInfo", {}).get("gcsOutputDirectory", "")
        path_dir = output_dir.split("gs://")[1]
        bucket_result_dir, output_dir = path_dir.split("/")
        bucket_result = storage_client.get_bucket(bucket_result_dir)
        blobs = list(bucket_result.list_blobs(prefix=f"{output_dir}/predictions"))
        if len(blobs) == 0:
            raise ProviderException("No result found !!")
        blob_result = blobs[0]
        result = json.loads(blob_result.download_as_string())

        standardized_response = AutomlClassificationPrediction(
            classes=result["prediction"]["displayNames"],
            confidences=result["prediction"]["confidences"],
        )
        return AsyncResponseType[AutomlClassificationPrediction](
            original_response=result["prediction"],
            standardized_response=standardized_response,
            provider_job_id=prediction_job_id,
        )
