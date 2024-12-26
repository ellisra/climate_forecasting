from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAXModel:
    """SARIMAX model class containing model functions

    Attributes:
        feature: Feature to be trained on/forecasted
        endog_train: Time series of feature
        exog_train: Time series of exogenous features
        exog_future: Fututre values for exogenous time series
        order: pdq values
        seasonal_order: pdq values with seasonal component
        freq: Frequency of time series steps
        model: SARIMAX model
        model_fit: SARIMAX model trained on dataset
        forecasts: Time series forecasts produced by the model
    """

    def __init__(
        self,
        feature: str,
        endog_train: pd.DataFrame,
        exog_train: pd.DataFrame,
        exog_future: pd.DataFrame,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        freq: str = "d",
    ):
        self.feature = feature
        self.endog_train = endog_train
        self.exog_train = exog_train
        self.exog_future = exog_future
        self.order = order
        self.seasonal_order = seasonal_order
        self.freq = freq

    def train(self) -> None:
        """Defines and fits the model to the historical time series"""
        self.model = SARIMAX(
            endog=self.endog_train[self.feature],
            exog=self.exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            freq=self.freq,
        )

        self.model_fit = self.model.fit()

    def forecast(
        self,
        forecast_path: Path,
        output_name: str,
        n_steps: int,
    ) -> None:
        """Forecasts and saves the next n steps in the time series

        Args:
            forecast_path: Path to save forecasts to
            output_name: Name for saved file
            n_steps: Number of steps to forecast
        """
        self.forecasts = self.model_fit.forecast(  # type: ignore
            steps=n_steps,
            exog=self.exog_future,
        )
        self.forecasts.to_csv(Path(f"{forecast_path}/{output_name}.csv"))


# def run(
#     data_path: Path,
#     forecast_path: Path,
#     model_path: Path,
#     order: tuple[int, int, int],
#     seasonal_order: tuple[int, int, int, int],
#     n_steps: int,
#     freq: str = "d",
# ) -> None:
#     """Defines, fits, and saves models as well as their forecasts for the next n_steps
#
#     Args:
#         data_path: Path to training data
#         forecast_path: Where to save forecasts
#         model_path: Where to save models
#         order: Autoregressive, differencing, and moving averages
#         seasonal_order: Order with seasonal period of the data
#         n_steps: Number of steps to forecast
#         freq: Frequency of time series steps
#     """
#     pass
