import logging

import joblib
import typer

from src.modeling import optimize_hp, build_model
from src.processor import Processor

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command(
    help="Upload data to the database."
)
def insert_data_to_db() -> None:
    processor = Processor()
    data = processor.read_data()
    processor.process(data)
    typer.echo("Data has been uploaded to the database.")


@app.command(
    help="Prepare data for modeling."
)
def process_data() -> None:
    df = Processor().read_data()
    Processor().process(df)
    typer.echo("Data preparation is complete.")


@app.command(
    help="Train the XGBoost model, optimize hyperparameters, and save the best model."
)
def train_model(output_model_path: str = "models/best_xgb.pkl"):
    """Train the XGBoost model, optimize hyperparameters, and save the best model."""
    typer.echo("Data loading.")
    processor = Processor()
    raw_data = processor.read_data()
    processed_data = processor.process(raw_data)
    typer.echo("Optimizing HP.")
    best_trial = optimize_hp(processed_data)
    typer.echo("Training best model on full data.")
    X_train, y_train, _, _ = processor.split_data(processed_data)
    best_model = build_model(trial=best_trial)  # Use full parameter set
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, output_model_path)
    typer.echo(f"Best model saved to {output_model_path}")


if __name__ == "__main__":
    app()
