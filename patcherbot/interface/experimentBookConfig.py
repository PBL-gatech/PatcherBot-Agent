"""Configuration values displayed by the Experiment Book tab."""

import logging

import param

from patcherbot.utils.config import Config


class ExperimentBookConfig(Config):
    """Runtime metadata for the current Experiment Book session."""

    experiment_name = param.String(default="", doc="Experiment name")
    strain_culture = param.String(default="", doc="Strain/Culture")
    gender = param.String(default="", doc="Gender")
    age = param.String(default="", doc="Age")
    general_notes = param.String(default="", doc="General notes")

    categories = [
        (
            "Experiment Details",
            ["experiment_name", "strain_culture", "gender", "age"],
        ),
        ("Notes", ["general_notes"]),
    ]

    logging.info("ExperimentBookConfig initialized successfully.")
