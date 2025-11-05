import base64
import json
import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from typing import Sequence

import pytest
import requests
from PIL import Image

from edenai_apis.apis.google.google_image_api import GoogleImageApi
from edenai_apis.features.image.embeddings.embeddings_dataclass import (
    EmbeddingsDataClass,
    EmbeddingDataClass,
)
from edenai_apis.utils.types import ResponseType
from edenai_apis.utils.exception import ProviderException


@pytest.fixture
def sample_image_file():
    """Create a temporary sample image file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        # Create a simple 100x100 RGB image
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(tmp_file, format="PNG")
        tmp_file.flush()
        yield tmp_file.name
    # Cleanup
    os.unlink(tmp_file.name)


@pytest.fixture
def mock_google_response():
    """Mock response from Google AI Platform API."""
    mock_embedding = [0.2] * 1408  # Default 1408-dimensional embedding
    mock_response = {
        "predictions": [{"imageEmbedding": mock_embedding}]
    }
    return mock_response


@pytest.fixture
def google_image_api():
    """Create an instance of GoogleImageApi with mocked clients."""
    # Create instance without calling __init__
    api = GoogleImageApi.__new__(GoogleImageApi)
    api.project_id = "test-project-id"
    api.location = "us-central1"
    api.clients = {
        "image": MagicMock(),
        "llm_client": MagicMock(),
    }
    return api


@pytest.mark.google
class TestGoogleImageEmbeddingsSync:
    """Test suite for synchronous image__embeddings method."""

    def test_embeddings_success(
        self, google_image_api, sample_image_file, mock_google_response
    ):
        """Test successful embedding generation with valid image file."""
        # Mock requests.post
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_google_response
            mock_post.return_value = mock_response_obj

            # Call the method
            result = google_image_api.image__embeddings(
                file=sample_image_file,
                model="multimodalembedding@001",
                embedding_dimension=1408,
            )

            # Assertions
            assert isinstance(result, ResponseType)
            assert isinstance(result.standardized_response, EmbeddingsDataClass)
            assert len(result.standardized_response.items) == 1
            assert isinstance(result.standardized_response.items[0], EmbeddingDataClass)
            assert len(result.standardized_response.items[0].embedding) == 1408
            assert result.original_response["predictions"][0]["imageEmbedding"] == [0.2] * 1408

            # Verify the API was called
            mock_post.assert_called_once()

    def test_embeddings_with_dimension_128(
        self, google_image_api, sample_image_file
    ):
        """Test embedding generation with 128 dimensions."""
        mock_embedding = [0.2] * 128
        mock_response = {"predictions": [{"imageEmbedding": mock_embedding}]}

        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj

            result = google_image_api.image__embeddings(
                file=sample_image_file,
                embedding_dimension=128,
            )

            # Verify the request includes the correct dimension
            call_args = mock_post.call_args
            request_body = call_args[1]["json"]
            assert request_body["parameters"]["dimension"] == 128
            assert len(result.standardized_response.items[0].embedding) == 128

    def test_embeddings_with_dimension_256(
        self, google_image_api, sample_image_file
    ):
        """Test embedding generation with 256 dimensions."""
        mock_embedding = [0.2] * 256
        mock_response = {"predictions": [{"imageEmbedding": mock_embedding}]}

        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj

            result = google_image_api.image__embeddings(
                file=sample_image_file,
                embedding_dimension=256,
            )

            # Verify the request includes the correct dimension
            call_args = mock_post.call_args
            request_body = call_args[1]["json"]
            assert request_body["parameters"]["dimension"] == 256
            assert len(result.standardized_response.items[0].embedding) == 256

    def test_embeddings_with_dimension_512(
        self, google_image_api, sample_image_file
    ):
        """Test embedding generation with 512 dimensions."""
        mock_embedding = [0.2] * 512
        mock_response = {"predictions": [{"imageEmbedding": mock_embedding}]}

        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj

            result = google_image_api.image__embeddings(
                file=sample_image_file,
                embedding_dimension=512,
            )

            # Verify the request includes the correct dimension
            call_args = mock_post.call_args
            request_body = call_args[1]["json"]
            assert request_body["parameters"]["dimension"] == 512
            assert len(result.standardized_response.items[0].embedding) == 512

    def test_embeddings_with_dimension_1408(
        self, google_image_api, sample_image_file, mock_google_response
    ):
        """Test embedding generation with 1408 dimensions (default)."""
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_google_response
            mock_post.return_value = mock_response_obj

            result = google_image_api.image__embeddings(
                file=sample_image_file,
                embedding_dimension=1408,
            )

            # Verify the request includes the correct dimension
            call_args = mock_post.call_args
            request_body = call_args[1]["json"]
            assert request_body["parameters"]["dimension"] == 1408
            assert len(result.standardized_response.items[0].embedding) == 1408

    def test_embeddings_invalid_file(self, google_image_api):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            google_image_api.image__embeddings(
                file="/path/to/nonexistent/file.png",
                embedding_dimension=1408,
            )

    def test_embeddings_api_error_in_response(
        self, google_image_api, sample_image_file
    ):
        """Test handling of API error in response."""
        mock_error_response = {
            "error": {"message": "Invalid image format", "code": 400}
        }

        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_error_response
            mock_post.return_value = mock_response_obj

            with pytest.raises(ProviderException) as exc_info:
                google_image_api.image__embeddings(
                    file=sample_image_file,
                    embedding_dimension=1408,
                )

            assert "Invalid image format" in str(exc_info.value)

    def test_embeddings_no_predictions(
        self, google_image_api, sample_image_file
    ):
        """Test handling when no predictions are returned."""
        mock_response = {"predictions": []}

        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj

            with pytest.raises(ProviderException) as exc_info:
                google_image_api.image__embeddings(
                    file=sample_image_file,
                    embedding_dimension=1408,
                )

            assert "No predictions found" in str(exc_info.value)

    def test_embeddings_json_decode_error(
        self, google_image_api, sample_image_file
    ):
        """Test handling of JSON decode error."""
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.side_effect = json.JSONDecodeError("error", "", 0)
            mock_post.return_value = mock_response_obj

            with pytest.raises(ProviderException) as exc_info:
                google_image_api.image__embeddings(
                    file=sample_image_file,
                    embedding_dimension=1408,
                )

            assert "Internal Server Error" in str(exc_info.value)

    def test_embeddings_base64_encoding(
        self, google_image_api, sample_image_file, mock_google_response
    ):
        """Test that image is properly base64 encoded."""
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_google_response
            mock_post.return_value = mock_response_obj

            # Read the file to get expected base64
            with open(sample_image_file, "rb") as f:
                expected_b64 = base64.b64encode(f.read()).decode("utf-8")

            result = google_image_api.image__embeddings(
                file=sample_image_file,
                embedding_dimension=1408,
            )

            # Verify the request body contains correct base64 encoded image
            call_args = mock_post.call_args
            request_body = call_args[1]["json"]
            assert request_body["instances"][0]["image"]["bytesBase64Encoded"] == expected_b64

    def test_embeddings_response_structure(
        self, google_image_api, sample_image_file, mock_google_response
    ):
        """Test that response structure matches expected dataclass."""
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_google_response
            mock_post.return_value = mock_response_obj

            result = google_image_api.image__embeddings(
                file=sample_image_file,
                embedding_dimension=1408,
            )

            # Validate response structure
            assert hasattr(result, "original_response")
            assert hasattr(result, "standardized_response")
            assert isinstance(result.standardized_response.items, Sequence)
            assert all(
                isinstance(item, EmbeddingDataClass)
                for item in result.standardized_response.items
            )

    def test_embeddings_api_url_construction(
        self, google_image_api, sample_image_file, mock_google_response
    ):
        """Test that API URL is correctly constructed."""
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_google_response
            mock_post.return_value = mock_response_obj

            result = google_image_api.image__embeddings(
                file=sample_image_file,
                model="multimodalembedding@001",
                embedding_dimension=1408,
            )

            # Verify URL is correct
            call_args = mock_post.call_args
            expected_url = (
                "https://us-central1-aiplatform.googleapis.com/v1/"
                "projects/test-project-id/locations/us-central1/"
                "publishers/google/models/multimodalembedding@001:predict"
            )
            assert call_args[0][0] == expected_url

    def test_embeddings_authorization_header(
        self, google_image_api, sample_image_file, mock_google_response
    ):
        """Test that authorization header is correctly set."""
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-access-token-12345"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_google_response
            mock_post.return_value = mock_response_obj

            result = google_image_api.image__embeddings(
                file=sample_image_file,
                embedding_dimension=1408,
            )

            # Verify headers
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-access-token-12345"
            assert headers["Content-Type"] == "application/json"


@pytest.mark.google
@pytest.mark.asyncio
class TestGoogleImageEmbeddingsAsync:
    """Test suite for asynchronous image__aembeddings method."""

    @pytest.mark.asyncio
    async def test_aembeddings_with_file(
        self, google_image_api, sample_image_file, mock_google_response
    ):
        """Test async embedding generation with file path."""
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_google_response
            mock_post.return_value = mock_response_obj

            result = await google_image_api.image__aembeddings(
                file=sample_image_file,
                embedding_dimension=1408,
            )

            # Assertions
            assert isinstance(result, ResponseType)
            assert isinstance(result.standardized_response, EmbeddingsDataClass)
            assert len(result.standardized_response.items) == 1
            assert len(result.standardized_response.items[0].embedding) == 1408

    @pytest.mark.asyncio
    async def test_aembeddings_with_url(self, google_image_api, sample_image_file):
        """Test async embedding generation with file URL."""
        mock_embedding = [0.2] * 1408
        mock_response = {"predictions": [{"imageEmbedding": mock_embedding}]}

        # Mock FileHandler
        with patch("edenai_apis.apis.google.google_image_api.FileHandler") as mock_handler, \
             patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_post.return_value = mock_response_obj

            # Mock file download
            mock_file_wrapper = MagicMock()
            with open(sample_image_file, "rb") as f:
                file_content = f.read()
            mock_file_wrapper.get_file_b64_content.return_value = base64.b64encode(
                file_content
            ).decode("utf-8")

            mock_handler_instance = AsyncMock()
            mock_handler_instance.download_file = AsyncMock(return_value=mock_file_wrapper)
            mock_handler.return_value = mock_handler_instance

            result = await google_image_api.image__aembeddings(
                file="",
                file_url="https://example.com/image.png",
                embedding_dimension=1408,
            )

            # Assertions
            assert isinstance(result, ResponseType)
            assert isinstance(result.standardized_response, EmbeddingsDataClass)
            assert len(result.standardized_response.items[0].embedding) == 1408
            mock_handler_instance.download_file.assert_called_once_with(
                "https://example.com/image.png"
            )

    @pytest.mark.asyncio
    async def test_aembeddings_api_error(
        self, google_image_api, sample_image_file
    ):
        """Test async error handling for API errors."""
        mock_error_response = {
            "error": {"message": "Invalid request", "code": 400}
        }

        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_error_response
            mock_post.return_value = mock_response_obj

            with pytest.raises(ProviderException) as exc_info:
                await google_image_api.image__aembeddings(
                    file=sample_image_file,
                    embedding_dimension=1408,
                )

            assert "Invalid request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_aembeddings_response_structure(
        self, google_image_api, sample_image_file, mock_google_response
    ):
        """Test async response structure validation."""
        with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
             patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

            mock_token.return_value = "test-token"
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_google_response
            mock_post.return_value = mock_response_obj

            result = await google_image_api.image__aembeddings(
                file=sample_image_file,
                embedding_dimension=1408,
            )

            # Validate response structure
            assert hasattr(result, "original_response")
            assert hasattr(result, "standardized_response")
            assert isinstance(result.standardized_response, EmbeddingsDataClass)
            assert isinstance(result.standardized_response.items, Sequence)

    @pytest.mark.asyncio
    async def test_aembeddings_with_different_dimensions(
        self, google_image_api, sample_image_file
    ):
        """Test async embedding with different dimension sizes."""
        for dimension in [128, 256, 512, 1408]:
            mock_embedding = [0.2] * dimension
            mock_response = {"predictions": [{"imageEmbedding": mock_embedding}]}

            with patch("edenai_apis.apis.google.google_image_api.requests.post") as mock_post, \
                 patch("edenai_apis.apis.google.google_image_api.get_access_token") as mock_token:

                mock_token.return_value = "test-token"
                mock_response_obj = MagicMock()
                mock_response_obj.json.return_value = mock_response
                mock_post.return_value = mock_response_obj

                result = await google_image_api.image__aembeddings(
                    file=sample_image_file,
                    embedding_dimension=dimension,
                )

                assert len(result.standardized_response.items[0].embedding) == dimension
