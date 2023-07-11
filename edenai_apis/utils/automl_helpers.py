import os
from typing import Callable, Optional
import pandas as pd

from edenai_apis.utils.conversion import replace_sep

# MODULE NOT TESTED


# This function is temporary, it will be moved to google_automl when it is recreated
def google_specificities(file_dict: pd.DataFrame) -> pd.DataFrame:
    file_dict.assign(empty="")
    columns_title = ["empty", "docs", "labels"]
    file_dict = file_dict.reindex(columns=columns_title)


def format_csv_file_for_training(
    file_path: str,
    exit_path: str,
    provider_specificities: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
):
    file_dict = pd.read_csv(file_path)
    file_dict["labels"] = file_dict["labels"].apply(
        replace_sep, current_sep="|", new_sep=","
    )
    if provider_specificities:
        file_dict = provider_specificities(file_dict)
    file_dict.to_csv(exit_path, index=False, header=None)


def format_xml_automl_entry_file(
    provider_name: str,
    project_name: str,
    file_path: str,
    provider_specificities: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
):
    csv_output_dir = (
        f"{os.getcwd()}/media/data/automl-text/google/import/{provider_name}"
    )
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)
    csv_output_path = f"{csv_output_dir}/{project_name}"
    format_csv_file_for_training(file_path, csv_output_path, provider_specificities)
    return csv_output_path
