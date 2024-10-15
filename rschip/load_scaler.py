import pickle


def load_scaler(scaler_source):
    """Load a scaler's parameters from a provided source.

    This function accepts a source for the scaling parameters in the form of a dictionary
    or a string path to a pickle file. It returns a dictionary containing the scaler parameters.

    Args:
        scaler_source (dict or str): The source from which to load the scaler parameters.
            If a dictionary is provided, it is used directly. If a string is provided, it
            is treated as a path to a pickle file which is then loaded to retrieve the
            scaler dictionary.

    Returns:
        dict: A dictionary containing the scaling parameters.

    Raises:
        ValueError: If `scaler_source` is neither a dictionary nor a string pointing to
            a valid pickle file, or if the file does not exist.
    """
    # Load the scaling parameters
    if isinstance(scaler_source, dict):
        scaler_dict = scaler_source
    elif isinstance(scaler_source, str):
        try:
            with open(scaler_source, "rb") as f:
                scaler_dict = pickle.load(f)
        except FileNotFoundError:
            raise ValueError(f"The path {scaler_source} does not exist.")
    else:
        raise ValueError(
            "scaler_source must be a dictionary or a valid path to a pickle file."
        )
    return scaler_dict
