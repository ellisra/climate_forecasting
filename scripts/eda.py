from __future__ import annotations

import hydra
import pandas as pd
import plotly.graph_objects as go
from climate_forecasting.config import Config


@hydra.main(config_path="../", config_name="config", version_base=None)
def main(cfg: Config):
    """Main function for temporary exploratory data analysis snippets

    Args:
        cfg: Application configuration
    """
    df = pd.read_csv(cfg.paths.train_data).set_index("date")

    fig = go.Figure(data=go.Scatter(x=df.index, y=df["meantemp"]))
    fig.show()


if __name__ == "__main__":
    main()
