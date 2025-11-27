import base64
import json
import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from typing import Sequence

import pytest
from PIL import Image

from edenai_apis.apis.amazon.amazon_image_api import AmazonImageApi
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
        img = Image.new("RGB", (100, 100), color="red")
        img.save(tmp_file, format="PNG")
        tmp_file.flush()
        yield tmp_file.name
    # Cleanup
    os.unlink(tmp_file.name)


@pytest.fixture
def mock_bedrock_response():
    """Mock response from AWS Bedrock API."""
    mock_embedding = [0.1] * 256  # 256-dimensional embedding
    mock_response = {
        "body": MagicMock(read=lambda: json.dumps({"embedding": mock_embedding}).encode())
    }
    return mock_response


@pytest.fixture
def amazon_image_api():
    """Create an instance of AmazonImageApi with mocked clients."""
    # Create instance without calling __init__
    api = AmazonImageApi.__new__(AmazonImageApi)
    api.clients = {
        "bedrock": MagicMock(),
        "image": MagicMock(),
    }
    return api


@pytest.mark.amazon
class TestAmazonImageEmbeddingsSync:
    """Test suite for synchronous image__embeddings method."""

    def test_embeddings_success(self, amazon_image_api, sample_image_file, mock_bedrock_response):
        """Test successful embedding generation with valid image file."""
        # Mock the bedrock invoke_model call
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_bedrock_response
        )

        # Call the method
        result = amazon_image_api.image__embeddings(
            file=sample_image_file,
            model="titan-embed-image-v1",
            embedding_dimension=256,
        )

        # Assertions
        assert isinstance(result, ResponseType)
        assert isinstance(result.standardized_response, EmbeddingsDataClass)
        assert len(result.standardized_response.items) == 1
        assert isinstance(result.standardized_response.items[0], EmbeddingDataClass)
        assert len(result.standardized_response.items[0].embedding) == 256
        assert result.original_response["embedding"] == [0.1] * 256

        # Verify the bedrock client was called
        amazon_image_api.clients["bedrock"].invoke_model.assert_called_once()

    def test_embeddings_with_dimension_256(
        self, amazon_image_api, sample_image_file, mock_bedrock_response
    ):
        """Test embedding generation with 256 dimensions."""
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_bedrock_response
        )

        result = amazon_image_api.image__embeddings(
            file=sample_image_file,
            embedding_dimension=256,
        )

        # Verify the request includes the correct dimension
        call_args = amazon_image_api.clients["bedrock"].invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert request_body["embeddingConfig"]["outputEmbeddingLength"] == 256

    def test_embeddings_with_dimension_384(
        self, amazon_image_api, sample_image_file
    ):
        """Test embedding generation with 384 dimensions."""
        mock_embedding = [0.1] * 384
        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({"embedding": mock_embedding}).encode())
        }
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_response
        )

        result = amazon_image_api.image__embeddings(
            file=sample_image_file,
            embedding_dimension=384,
        )

        # Verify the request includes the correct dimension
        call_args = amazon_image_api.clients["bedrock"].invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert request_body["embeddingConfig"]["outputEmbeddingLength"] == 384
        assert len(result.standardized_response.items[0].embedding) == 384

    def test_embeddings_with_dimension_1024(
        self, amazon_image_api, sample_image_file
    ):
        """Test embedding generation with 1024 dimensions."""
        mock_embedding = [0.1] * 1024
        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({"embedding": mock_embedding}).encode())
        }
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_response
        )

        result = amazon_image_api.image__embeddings(
            file=sample_image_file,
            embedding_dimension=1024,
        )

        # Verify the request includes the correct dimension
        call_args = amazon_image_api.clients["bedrock"].invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert request_body["embeddingConfig"]["outputEmbeddingLength"] == 1024
        assert len(result.standardized_response.items[0].embedding) == 1024

    def test_embeddings_invalid_file(self, amazon_image_api):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            amazon_image_api.image__embeddings(
                file="/path/to/nonexistent/file.png",
                embedding_dimension=256,
            )

    def test_embeddings_base64_encoding(
        self, amazon_image_api, sample_image_file, mock_bedrock_response
    ):
        """Test that image is properly base64 encoded."""
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_bedrock_response
        )

        # Read the file to get expected base64
        with open(sample_image_file, "rb") as f:
            expected_b64 = base64.b64encode(f.read()).decode("utf-8")

        result = amazon_image_api.image__embeddings(
            file=sample_image_file,
            embedding_dimension=256,
        )

        # Verify the request body contains correct base64 encoded image
        call_args = amazon_image_api.clients["bedrock"].invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert request_body["inputImage"] == expected_b64

    def test_embeddings_response_structure(
        self, amazon_image_api, sample_image_file, mock_bedrock_response
    ):
        """Test that response structure matches expected dataclass."""
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_bedrock_response
        )

        result = amazon_image_api.image__embeddings(
            file=sample_image_file,
            embedding_dimension=256,
        )

        # Validate response structure
        assert hasattr(result, "original_response")
        assert hasattr(result, "standardized_response")
        assert isinstance(result.standardized_response.items, Sequence)
        assert all(
            isinstance(item, EmbeddingDataClass)
            for item in result.standardized_response.items
        )

    def test_embeddings_model_id_parameter(
        self, amazon_image_api, sample_image_file, mock_bedrock_response
    ):
        """Test that model parameter is correctly passed to API."""
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_bedrock_response
        )

        result = amazon_image_api.image__embeddings(
            file=sample_image_file,
            model="titan-embed-image-v1",
            embedding_dimension=256,
        )

        # Verify model ID is correct
        call_args = amazon_image_api.clients["bedrock"].invoke_model.call_args
        assert call_args[1]["modelId"] == "amazon.titan-embed-image-v1"


@pytest.mark.amazon
@pytest.mark.asyncio
class TestAmazonImageEmbeddingsAsync:
    """Test suite for asynchronous image__aembeddings method."""

    @pytest.mark.asyncio
    async def test_aembeddings_with_file(
        self, amazon_image_api, sample_image_file, mock_bedrock_response
    ):
        """Test async embedding generation with file path."""
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_bedrock_response
        )

        result = await amazon_image_api.image__aembeddings(
            file=sample_image_file,
            embedding_dimension=256,
        )

        # Assertions
        assert isinstance(result, ResponseType)
        assert isinstance(result.standardized_response, EmbeddingsDataClass)
        assert len(result.standardized_response.items) == 1
        assert len(result.standardized_response.items[0].embedding) == 256

    @pytest.mark.asyncio
    async def test_aembeddings_with_url(self, amazon_image_api, sample_image_file):
        """Test async embedding generation with file URL."""
        mock_embedding = [0.1] * 256
        mock_response = {
            "body": MagicMock(read=lambda: json.dumps({"embedding": mock_embedding}).encode())
        }
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_response
        )

        # Mock FileHandler
        with patch("edenai_apis.apis.amazon.amazon_image_api.FileHandler") as mock_handler:
            mock_file_wrapper = MagicMock()
            # Read actual file content for base64
            with open(sample_image_file, "rb") as f:
                file_content = f.read()
            mock_file_wrapper.get_file_b64_content.return_value = base64.b64encode(
                file_content
            ).decode("utf-8")

            mock_handler_instance = AsyncMock()
            mock_handler_instance.download_file = AsyncMock(return_value=mock_file_wrapper)
            mock_handler.return_value = mock_handler_instance

            result = await amazon_image_api.image__aembeddings(
                file="",
                file_url="https://example.com/image.png",
                embedding_dimension=256,
            )

            # Assertions
            assert isinstance(result, ResponseType)
            assert isinstance(result.standardized_response, EmbeddingsDataClass)
            assert len(result.standardized_response.items[0].embedding) == 256
            mock_handler_instance.download_file.assert_called_once_with(
                "https://example.com/image.png"
            )

    @pytest.mark.asyncio
    async def test_aembeddings_with_different_dimensions(
        self, amazon_image_api, sample_image_file
    ):
        """Test async embedding with different dimension sizes."""
        for dimension in [256, 384, 1024]:
            mock_embedding = [0.1] * dimension
            mock_response = {
                "body": MagicMock(
                    read=lambda e=mock_embedding: json.dumps({"embedding": e}).encode()
                )
            }
            amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
                return_value=mock_response
            )

            result = await amazon_image_api.image__aembeddings(
                file=sample_image_file,
                embedding_dimension=dimension,
            )

            assert len(result.standardized_response.items[0].embedding) == dimension

    @pytest.mark.asyncio
    async def test_aembeddings_response_structure(
        self, amazon_image_api, sample_image_file, mock_bedrock_response
    ):
        """Test async response structure validation."""
        amazon_image_api.clients["bedrock"].invoke_model = MagicMock(
            return_value=mock_bedrock_response
        )

        result = await amazon_image_api.image__aembeddings(
            file=sample_image_file,
            embedding_dimension=256,
        )

        # Validate response structure
        assert hasattr(result, "original_response")
        assert hasattr(result, "standardized_response")
        assert isinstance(result.standardized_response, EmbeddingsDataClass)
        assert isinstance(result.standardized_response.items, Sequence)
