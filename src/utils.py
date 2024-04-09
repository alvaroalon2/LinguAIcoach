from typing import Any, Dict


def convert_json_to_str(x: Dict[str, str]) -> str:
    """
    A function that converts a JSON object to a formatted string.

    Parameters:
        x (dict): A dictionary representing a JSON object.

    Returns:
        str: A formatted string with key-value pairs from the JSON object.
    """
    return " \n\n ".join(f"**{k}**" + ": " + val for k, val in x.items() if val != "")


def read_json(file_path: str) -> Dict[str, Any]:
    """
    A function that reads a JSON file and returns its contents as a dictionary.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the contents of the JSON file.
    """
    with open(file_path) as f:
        return dict(eval(f.read()))


def read_plain_text(file_path: str) -> str:
    """
    A function that reads a plain text file and returns its contents as a string.

    Parameters:
        file_path (str): The path to the plain text file.

    Returns:
        str: A string containing the contents of the plain text file.
    """
    with open(file_path) as f:
        return f.read()
