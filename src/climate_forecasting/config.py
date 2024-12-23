from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    """Paths to directories

    Attributes:
        train_data: Training data
        test_data: Testing data
        outputs: Location of model outputs
        probabilities: Location of probabilities returned by the model
        forecasts: Location of forecasts returned by the model
        models: Location of trained models
    """

    train_data: Path
    test_data: Path
    outputs: Path
    probabilities: Path
    forecasts: Path
    models: Path


class Params(BaseModel):
    """Parameters for use in training and forecasting

    Attributes:
        clear: Whether to clear previous outputs of the model
        n_steps: The number of steps to forecast
        feature: The feature to forecast or train on
        pdq: The number of autoregressive terms,
             the number of differences needed for stationarity,
             and the number of lagged forecast errors in the prediction equation
        seasonality: The number of steps per 'season' in the data
    """

    clear: bool
    n_steps: int
    feature: str
    pdq: tuple[int, int, int]
    seasonality: int


class Config(BaseModel):
    """Configuration of the application

    Attributes:
        paths: Paths to directories
        params: Tunable parameters
    """

    paths: Paths
    params: Params
