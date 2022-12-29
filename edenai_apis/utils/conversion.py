from builtins import bool
import os
import re
import datetime as dt
from typing import Optional, Type, Union
import pandas as pd

from edenai_apis.utils.exception import ProviderException
from edenai_apis.utils.public_enum import AutomlClassificationProviderName


def convert_string_to_number(
    string_number: Optional[str], val_type: Union[Type[int], Type[float]]
) -> Union[int, float, None]:
    """convert a `string` to either `int` or `float`"""
    if isinstance(string_number, (int, float)):
        return string_number
    if isinstance(string_number, str):
        string_number = string_number.strip()
    if not string_number:
        return None
    try:
        number = val_type(re.sub(r"[^\d\.]", "", string_number))
        return number
    except Exception as exc:
        return None


def retreive_first_number_from_string(string_number: str) -> Union[str, None]:
    """
    Find the first number found in a string
    Returns:
        str:    if found the number is returned as a string
        None:   if nothing is found
    """
    return re.findall(r"\d+", string_number)[0] if string_number is not None else None


def combine_date_with_time(date: Optional[str], time: Union[str, None]) -> Union[str, None]:
    """
    Concatenate date string and time string
    Returns:
        - `None`: if `date` or `time` is `None`
        - `str`: if concatenation is successful
    """
    if time is not None:
        for fmt in ["%H:%M", "%H:%M:%:%S"]:
            try:
                time = dt.datetime.strptime(time, fmt).time()
                date = (
                    str(
                        dt.datetime.combine(
                            dt.datetime.strptime(date, "%Y-%m-%d"), time
                        )
                    )
                    if date is not None
                    else None
                )
                return date
            except ValueError:
                pass
        return date
    return None


def convert_pt_date_to_string(pt_date: str):
    # dates = re.findall('PT(\d*)H{0,1}(\d*)M{0,1}(\d*)S{0,1}', pt_date)[0]
    date = pt_date.split("PT")[1]
    hours, date = date.split("H") if "H" in date else (0, date)
    minutes, date = date.split("M") if "M" in date else (0, date)
    seconds = float(date.split("S")[0] or 0.0) if "S" in date else 0 
    hours = int(hours or 0)
    minutes = int(minutes or 0)
    return 3600*hours + 60*minutes + seconds



def format_string_url_language(
    url: str, language: str, prefix_lang: str, provider_name: str, is_url: bool = True
):
    """
    Concatenates an url with a language code

    Args:
        url (str): the url string to be concatinated with the language code
        language (str): the language code value to concatenate with the url
        prefix_lang (str): the language code name prefix for the url
        provider_name (str): the provider name
        is_url (bool, optional): specifies if the url used is for a GET request. Defaults to True.

    Raises:
        ProviderException: throws a provider exception if the language in None

    Returns:
        str: the url formatted with the language code
    """
    if not language:
        return url
    if is_url:
        return (
            f"{url}&{prefix_lang}={language}"
            if "?" in url
            else f"{url}?{prefix_lang}={language}"
        )
    return f"{url}{prefix_lang}{language}"


def replace_sep(x):
    if isinstance(x, str):
        x = x.replace("|", ",")
        x = re.sub(r",$", "", x)
    return x


def format_csv_file_for_training(provider_name, file_path, exit_path):
    file_dict = pd.read_csv(file_path)
    file_dict["labels"] = file_dict["labels"].apply(replace_sep)
    if provider_name == AutomlClassificationProviderName.GOOGLE.value:
        file_dict.assign(empty="")
        columns_title = ["empty", "docs", "labels"]
        file_dict = file_dict.reindex(columns=columns_title)
        file_dict.to_csv(exit_path, index=False, header=None)
    else:
        file_dict.to_csv(exit_path, index=False, header=None)


def format_xml_automl_entry_file(provider_name, file_path, project_name):
    csv_output_dire = (
        f"{os.getcwd()}/media/data/automl-text/google/import/{provider_name}"
    )
    if not os.path.exists(csv_output_dire):
        os.makedirs(csv_output_dire)
    csv_output = f"{csv_output_dire}/{project_name}"
    format_csv_file_for_training(provider_name, file_path, csv_output)
    return csv_output
