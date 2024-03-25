from typing import Dict, List, Optional

from edenai_apis.loaders.data_loader import ProviderDataEnum
from edenai_apis.loaders.loaders import load_provider
from edenai_apis.utils.audio import get_file_extension, retreive_voice_id
from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.files import FileWrapper
from edenai_apis.utils.languages import (
    LanguageErrorMessage,
    provide_appropriate_language,
    load_standardized_language,
)
from edenai_apis.utils.resolutions import provider_appropriate_resolution


def validate_input_file_extension(constraints: dict, args: dict) -> dict:
    """Check that a provider offers support for the input file extension for speech to text

    Args:
        - constraints (dict): all constraints (on inputs) of the provider
        - args (dict): inputs passed to the provider call
        - provider (str): provider name

    Returns:
        - `args` (dict): same or updated args
    Raises:
        - `ProviderException`: if file extension is not supported or in the provider requires a certain number of audio channels
    """

    provider_file_extensions_constraints: List[str] = constraints.get(
        "file_extensions", []
    )

    input_file: Optional[FileWrapper] = args.get("file")

    if not input_file or not provider_file_extensions_constraints:
        return args
    export_format = get_file_extension(input_file, provider_file_extensions_constraints)
    frame_rate = input_file.file_info.file_frame_rate
    channels = input_file.file_info.file_channels
    args["audio_attributes"] = (export_format, channels, frame_rate)
    return args


def validate_resolution(constraints: dict, args: dict) -> dict:
    supported_resolutions: List[str] = constraints.get("resolutions", [])

    if not args.get("resolution") or not supported_resolutions:
        return args
    try:
        resolution = provider_appropriate_resolution(args["resolution"])
    except SyntaxError as exc:
        raise ProviderException(exc)

    data = resolution.split("x")
    if len(data) != 2:
        raise ProviderException(f"Invalid resolution format :`{args['resolution']}`.")

    if resolution not in supported_resolutions:
        raise ProviderException(
            f"Resolution not supported by the provider. Use one of the following resolutions: {','.join(supported_resolutions)}"
        )

    args["resolution"] = resolution
    return args


def validate_input_file_type(constraints: dict, provider: str, args: dict) -> dict:
    """Check that a provider offers support for the input file type

    Args:
        - constraints (dict): all constraints (on inputs) of the provider
        - args (dict): inputs passed to the provider call
        - provider (str): provider name

    Returns:
        - `args` (dict): same or updated args

    Raises:
        - `ProviderException`: if file is not supported
    """
    provider_file_type_constraints: List[str] = constraints.get("file_types", [])

    input_file: FileWrapper = args.get("file")

    if input_file and len(provider_file_type_constraints) > 0:
        input_file_type = input_file.file_info.file_media_type

        if input_file_type is None:
            # if mimetype is not recognized we don't validate it
            # eg: webp and raw images are not recognized but are accepted by google ocr
            return args

        # constraint can be written as "image/*" for example
        # it means it accepts all types of images
        type_glob = [
            constraint.split("/")[0]
            for constraint in provider_file_type_constraints
            if constraint.endswith("*")
        ]

        if input_file_type not in provider_file_type_constraints and not any(
            [global_type in input_file_type for global_type in type_glob]
        ):
            supported_types = ",\n".join(provider_file_type_constraints)
            raise ProviderException(
                f"Provider {provider} doesn't support file type: {input_file_type} "
                f"for this feature.\n"
                f"Supported mimetypes are {supported_types}"
            )
    return args


def validate_single_language(
    provider_name: str,
    feature,
    subfeature,
    language: dict,
    null_language_accepted: bool,
) -> Optional[str]:
    """
    Validate and format given language

    Args:
        - provider_name (str)
        - feature (str): feature name
        - subfeature (str): subfeature name
        - language (dict): Dictionnary with `key` and `value` of input language. `value` canbe None. (ex: { 'key': 'source_langue', 'value': 'en'})
        - null_language_accepted (bool): if Provider can auto-detect langauages (accepts providing None as language)

    Returns:
        - language (str | None) validated language, can be None if provider accepts it

    Raises:
        - `ProviderException`: if language is not supported or cannot be None
    """

    # if user specifies the auto-detect language
    if language["value"] and language["value"].lower() == "auto-detect":
        language["value"] = None

    if not language["value"]:
        if null_language_accepted is True:
            return language["value"]
        else:
            raise ProviderException(
                LanguageErrorMessage.LANGUAGE_REQUIRED(language["key"])
            )

    try:
        formated_language = provide_appropriate_language(
            language["value"],
            provider_name=provider_name,
            feature=feature,
            subfeature=subfeature,
        )
    except SyntaxError as exc:
        raise ProviderException(
            LanguageErrorMessage.LANGUAGE_SYNTAX_ERROR(language["value"])
        )

    if null_language_accepted is False:
        if formated_language is None:
            if "-" in language["value"]:
                supported_languages = load_standardized_language(
                    feature, subfeature, [provider_name]
                )
                suggested_language = language["value"].split("-")[0]
                if suggested_language in supported_languages:
                    raise ProviderException(
                        LanguageErrorMessage.LANGUAGE_GENERIQUE_REQUESTED(
                            language["value"], suggested_language, language["key"]
                        )
                    )
            raise ProviderException(
                LanguageErrorMessage.LANGUAGE_NOT_SUPPORTED(
                    language["value"], language["key"]
                )
            )

    return formated_language


def validate_all_input_languages(
    constraints: dict, args: dict, provider_name: str, feature: str, subfeature: str
) -> Dict:
    """
    Updates the args input to provide the appropriate language
    supported by the provider from user input language,
    while checking at the same time if None language is acceptable

    Args:
        - constraints: provider constraints for this subfeature
        - args (dict): feature arguments, exp: {text: 'alleluia', language='en'}
        - provider_name (str): the provider name, exp: Google
        - feature (str): the feature name, exp: text
        - subfeature (str): the subfeature name, exp: sentiment_analysis

    Returns:
        - dict: updated args
    """

    # Skip language checking for text_to_speech if settings are passed, execpt for google
    if (
        subfeature == "text_to_speech"
        and provider_name in (args.get("settings", {}) or {})
        and provider_name != "google"
    ):
        return args

    accepts_null_language = constraints.get("allow_null_language", False)

    for argument_name, argument_value in args.items():
        if "language" not in argument_name:
            continue

        args[argument_name] = validate_single_language(
            provider_name=provider_name,
            feature=feature,
            subfeature=subfeature,
            language={"key": argument_name, "value": argument_value},
            null_language_accepted=accepts_null_language,
        )
    return args


def validate_audio_format(constraints: dict, args: dict) -> dict:
    provider_audio_format_constraints: List[str] = (
        constraints.get("audio_format", []) or []
    )

    audio_format = args.get("audio_format")

    if audio_format and audio_format not in provider_audio_format_constraints:
        raise ProviderException(
            f"Audio format not supported. Use one of the following: {', '.join(provider_audio_format_constraints)}"
        )

    return args


def validate_models(
    provider: str, subfeature: str, constraints: dict, args: dict
) -> Dict:
    voice_ids = constraints.get("voice_ids")
    settings = args.get("settings", {})

    if "text_to_speech" in subfeature and voice_ids:
        if any(option in voice_ids for option in ["MALE", "FEMALE"]):
            voice_id = retreive_voice_id(
                provider, subfeature, args["language"], args["option"], settings
            )
            args["voice_id"] = voice_id
    else:
        if settings and provider in settings:
            selected_model = settings[provider]
            args["model"] = selected_model

    args.pop("settings", None)
    return args


def validate_document_type(subfeature: str, constraints: dict, args: dict) -> Dict:
    """
    Validate document type based on specified constraints.

    Parameters:
    - subfeature (str): The subfeature being validated.
    - constraints (dict): Constraints for document validation.
    - args (dict): Arguments containing document details.

    Returns:
    - Dict: Validated arguments.
    """
    if subfeature == "financial_parser":
        documents = constraints.get("documents")

        # If no documents are specified, return the arguments as is
        if not documents:
            return args

        # Handle the case where null document type is allowed
        if (
            constraints.get("allow_null_document_type")
            and args.get("document_type") == "auto-detect"
        ):
            args["document_type"] = ""
            return args

        # Check if the document type is allowed or raise an exception
        if (
            not constraints.get("allow_null_document_type")
            and args["document_type"] == "auto-detect"
        ):
            raise ProviderException(
                "The provider does not accept auto-detect for this feature."
            )

        # Return the validated arguments
        return args
    else:
        # If the subfeature is not financial_parser, return the arguments as is
        return args


def transform_file_args(args: dict) -> dict:
    """transform the file wrapper to file path and file url for subfeature functions

    Args:
        args (dict): feature arguments, exp: {text: 'alleluia', language='en'}

    Returns:
        dict: updated args
    """
    file_args = ["file", "file1", "file2"]

    for file_arg in file_args:
        if args.get(file_arg) and isinstance(args.get(file_arg), FileWrapper):
            file_wrapper: FileWrapper = args[file_arg]
            file_path = file_wrapper.file_path
            file_url = file_wrapper.file_url
            args.update({file_arg: file_path, f"{file_arg}_url": file_url})
    if args.get("files"):
        files = []
        files_url = []
        for file in args.get("files", []):
            if isinstance(file, FileWrapper):
                file_wrapper: FileWrapper = file
                file_path = file_wrapper.file_path
                file_url = file_wrapper.file_url
                files.append(file_path)
                files_url.append(file_url)
        args.update({"files": files, "files_url": files_url})

    return args


def validate_all_provider_constraints(
    provider: str, feature: str, subfeature: str, phase: str, args: dict
) -> dict:
    """
    Validate inputs arguments against provider constraints

    Args:
        - provider (str): provider name
        - feature (str): feature name
        - subfeature (str): subfeature name
        - args (dict): dictionnary of input arguments

    Returns:
        - args: updated/validated args
    """

    # load provider constraints
    provider_info = load_provider(
        ProviderDataEnum.PROVIDER_INFO,
        provider_name=provider,
        feature=feature,
        subfeature=subfeature,
        phase=phase,
    )
    provider_constraints = provider_info.get("constraints")

    if provider_constraints is not None:
        validated_args = args.copy()
        ## Validate here

        # file types
        validated_args = validate_input_file_type(
            provider_constraints, provider, validated_args
        )

        # languages
        validated_args = validate_all_input_languages(
            provider_constraints, validated_args, provider, feature, subfeature
        )

        # file extensions for audio files
        validated_args = validate_input_file_extension(
            provider_constraints, validated_args
        )
        # resolution for image generation
        validated_args = validate_resolution(provider_constraints, validated_args)

        # Audio format for text to speech
        validated_args = validate_audio_format(provider_constraints, validated_args)

        #  Validate models
        validated_args = validate_models(
            provider, subfeature, provider_constraints, validated_args
        )

        # Validate document_type
        validated_args = validate_document_type(
            subfeature, provider_constraints, validated_args
        )

        # ...

        validated_args = transform_file_args(validated_args)

        return validated_args

    args = validate_models(provider, subfeature, {}, args)

    args = transform_file_args(args)

    return args
