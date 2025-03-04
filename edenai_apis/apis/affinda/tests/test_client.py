import os

import pytest

from edenai_apis.apis.affinda.client import Client, ExtractorType
from edenai_apis.apis.affinda.models import Collection, Organization, Workspace
from edenai_apis.loaders.loaders import load_provider, ProviderDataEnum
from edenai_apis.utils.exception import ProviderException


@pytest.mark.skipif(
    os.environ.get("TEST_SCOPE") == "CICD-OPENSOURCE",
    reason="Skip in opensource package cicd workflow",
)
@pytest.mark.affinda
class TestClient:
    def setup_class(self) -> None:
        self.settings = load_provider(ProviderDataEnum.KEY, provider_name="affinda")
        self.client = Client(self.settings["api_key"])

    def test_get_organizations(self):
        organizations = self.client.get_organizations()

        assert isinstance(organizations, list)
        assert all(
            [isinstance(organization, Organization) for organization in organizations]
        )

    def test_get_organization__with_good_id(self):
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]

        organization = self.client.get_organization(edenai_organization.identifier)

        assert isinstance(organization, Organization)
        assert organization == edenai_organization

    def test_get_organization__with_bad_id(self):
        with pytest.raises(
            expected_exception=ProviderException,
            match="404 Client Error: Not Found for url:",
        ):
            organization = self.client.get_organization("bad_id__dfkredfmlmdfd")

    def test_current_organization_property(self):
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]

        assert (
            self.client.current_organization is None
        ), "current_organization must be None before any assigment"

        # Test current_organization.setter
        self.client.current_organization = edenai_organization.identifier
        assert (
            self.client.current_organization is not None
        ), "current_organization cannot be None after assigment"
        assert self.client.current_organization == edenai_organization

        # Test current_organization.deletter
        del self.client.current_organization
        assert (
            self.client.current_organization is None
        ), "current_organization must be None after deletion"

    def test_get_workspaces(self):
        # Test before assigment the organization
        with pytest.warns(
            expected_warning=Warning,
            match="You didn't have set a current_organization. Please see the documentations.",
        ):
            workspaces = self.client.get_workspaces()
            assert (
                workspaces == []
            ), "get_workspaces must be return an empty list if current_organization doesn't set"

        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Test if get_organizations correctly works
        workspaces = self.client.get_workspaces()

        # Clean client class
        del self.client.current_organization

        # Look the type of the returns
        assert isinstance(workspaces, list)
        assert all([isinstance(workspace, Workspace) for workspace in workspaces])

    def test_get_workspace__with_good_id(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Test if get_organizations correctly works
        workspace = self.client.get_workspaces()[0]

        specific_workspace = self.client.get_workspace(identifier=workspace.identifier)

        assert specific_workspace == workspace

    def test_get_workspace__with_bad_id(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        with pytest.raises(
            expected_exception=ProviderException,
            match="404 Client Error: Not Found for url:",
        ):
            workspace = self.client.get_workspace("bad_id__dfkredfmlmdfd")

    def test_current_workspace_property(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        assert (
            self.client.current_workspace is None
        ), "current_workspace must None before assigment"

        # Test current_workspace.setter
        workspace = self.client.get_workspaces()[0]
        self.client.current_workspace = workspace.identifier

        assert workspace is not None, "current_workspace cannot be None after assigment"
        assert self.client.current_workspace == workspace

        # Test current_workspace.deletter
        del self.client.current_workspace

        assert (
            self.client.current_workspace is None
        ), "current_workspace must be None after deletion"

    def test_create_and_delete_workspace(self):
        # Test before set current_organization
        del self.client.current_organization
        with pytest.raises(
            expected_exception=AttributeError,
            match="You didn't have set a current_organization. Please see the documentations.",
        ):
            self.client.create_workspace(name="Test-Workspace")

        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        new_workspace = self.client.create_workspace(
            name="Test-Workspace", visibility="organization"
        )

        # Delete the new workspace
        self.client.delete_workspace(identifier=new_workspace.identifier)

        # Look the attribute of the new workspace
        assert new_workspace.name == "Test-Workspace"
        assert new_workspace.visibility == "organization"

        # Check if the new_workspace has been deleted
        with pytest.raises(
            expected_exception=ProviderException,
            match="404 Client Error: Not Found for url:",
        ):
            self.client.get_workspace(identifier=new_workspace.identifier)

    def test_get_collections(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Test before assigment workspace
        with pytest.warns(
            expected_warning=Warning,
            match="You didn't have set a current_workspace. Please see the documentations.",
        ):
            collections = self.client.get_collections()
            assert (
                collections == []
            ), "get_collections must be return an empty list when called before current_workspace's assigment"

        # Setup Workspace
        workspaces = self.client.get_workspaces()
        invoice_workspace = [
            workspace
            for workspace in workspaces
            if workspace.name == "Accounts Payable"
        ][0]
        self.client.current_workspace = invoice_workspace.identifier

        collections = self.client.get_collections()

        assert isinstance(collections, list)
        assert all([isinstance(collection, Collection) for collection in collections])

    def test_get_collection__with_good_id(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Setup Workspace
        workspaces = self.client.get_workspaces()
        invoice_workspace = [
            workspace
            for workspace in workspaces
            if workspace.name == "Accounts Payable"
        ][0]
        self.client.current_workspace = invoice_workspace.identifier

        collection = self.client.get_collections()[0]

        specific_collection = self.client.get_collection(
            identifier=collection.identifier
        )

        assert specific_collection == collection

    def test_get_collection__with_bad_id(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Setup Workspace
        workspaces = self.client.get_workspaces()
        invoice_workspace = [
            workspace
            for workspace in workspaces
            if workspace.name == "Accounts Payable"
        ][0]
        self.client.current_workspace = invoice_workspace.identifier

        with pytest.raises(
            expected_exception=ProviderException,
            match="404 Client Error: Not Found for url:",
        ):
            self.client.get_collection("bad_id_dksacksdncvdsnk")

    def test_current_collection_property(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Setup Workspace
        workspaces = self.client.get_workspaces()
        invoice_workspace = [
            workspace
            for workspace in workspaces
            if workspace.name == "Accounts Payable"
        ][0]
        self.client.current_workspace = invoice_workspace.identifier

        collection = self.client.get_collections()[0]

        assert (
            self.client.current_collection is None
        ), "current_collection must be None before assigment"

        # Test current_collection.setter
        self.client.current_collection = collection.identifier
        assert (
            self.client.current_collection is not None
        ), "current_collection cannot be None after assigment"
        assert self.client.current_collection == collection

        # Test current_collection.deletter
        del self.client.current_collection
        assert (
            self.client.current_collection is None
        ), "current_collection muts be None after deletion"

    def test_create_and_delete_collection(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Test before set current_workspace
        del self.client.current_workspace
        with pytest.raises(
            expected_exception=AttributeError,
            match="You didn't have set a current_workspace. Please see the documentations.",
        ):
            self.client.create_collection(
                name="Test-collection", extractor=ExtractorType.RESUME
            )

        # Setup Workspace
        workspaces = self.client.get_workspaces()
        invoice_workspace = [
            workspace
            for workspace in workspaces
            if workspace.name == "Accounts Payable"
        ][0]
        self.client.current_workspace = invoice_workspace.identifier

        # Create a new collection
        new_collection = self.client.create_collection(
            name="Test-collection", extractor=ExtractorType.RESUME
        )

        # Delete the new collection
        self.client.delete_collection(identifier=new_collection.identifier)

        assert new_collection.name == "Test-collection"
        assert new_collection.extractor.identifier == "resume"

        # Check if the new_collection has been deleted
        with pytest.raises(
            expected_exception=ProviderException,
            match="404 Client Error: Not Found for url:",
        ):
            self.client.get_workspace(identifier=new_collection.identifier)

    def test_get_documents(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Setup Workspace
        workspaces = self.client.get_workspaces()
        invoice_workspace = [
            workspace
            for workspace in workspaces
            if workspace.name == "Accounts Payable"
        ][0]
        self.client.current_workspace = invoice_workspace.identifier

        # Test before set the current_collection
        documents = self.client.get_documents()
        assert isinstance(documents, list)
        assert all([isinstance(document, dict) for document in documents])

        # Setup collection
        collection = self.client.get_collections()[0]
        self.client.current_collection = collection.identifier

        # Test get_documents
        documents = self.client.get_documents()
        assert isinstance(documents, list)
        assert all([isinstance(document, dict) for document in documents])

    def test_get_document_with_good_id(self):
        # Setup organizations
        organizations = self.client.get_organizations()
        edenai_organization = [
            orga for orga in organizations if orga.name == "Eden AI"
        ][0]
        self.client.current_organization = edenai_organization.identifier

        # Setup Workspace
        workspaces = self.client.get_workspaces()
        invoice_workspace = [
            workspace
            for workspace in workspaces
            if workspace.name == "Accounts Payable"
        ][0]
        self.client.current_workspace = invoice_workspace.identifier

        # Setup collection
        collection = self.client.get_collections()[0]
        self.client.current_collection = collection.identifier

        all_documents = self.client.get_documents()

        if len(all_documents):
            document = self.client.get_documents()[0]
        else:
            pytest.skip("No document exists")

        specific_document = self.client.get_document(
            identifier=document["meta"]["identifier"]
        )

        assert isinstance(specific_document, dict)
        assert specific_document["meta"]["identifier"] == document["meta"]["identifier"]

    def test_get_document_with_bad_id(self):
        with pytest.raises(
            expected_exception=ProviderException,
            match="404 Client Error: Not Found for url:",
        ):
            self.client.get_document(identifier="bad_id_fedlnswkfcjnsedl")
