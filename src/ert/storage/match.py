from __future__ import annotations

from abc import ABC, abstractmethod

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Union, cast, TypeVar, Generic, overload, Literal

import numpy as np
import polars as pl

from ert.config._reservoir_data import SeismicData
from ert.config.gen_data_config import GenDataConfig
from ert.config.observation_quality_control import (
    append_to_qc_error,
    ensure_qc_error_column,
    qc_rft_observations,
    qc_seismic_observations,
)

from ert.config.rft_config import RFTConfig
from ert.config.response_config import ResponseConfig
from ert.config.summary_config import SummaryConfig
from ert.config.seismic_config import SeismicConfig
from ert.config.breakthrough_config import BreakthroughConfig

if TYPE_CHECKING:
    import numpy.typing as npt

    from .local_ensemble import LocalEnsemble

logger = logging.getLogger(__name__)


class ReservoirDataMatchStrategy[ConcreteResponseConfig: ResponseConfig](ABC):
    ensemble: LocalEnsemble
    response_cls: ConcreteResponseConfig

    def __init__(self, ensemble: LocalEnsemble, response_cls: ConcreteResponseConfig) -> None:
        self.ensemble = ensemble
        self.response_cls = response_cls

    def prepare_raw_observations(self, raw_observations: pl.DataFrame) -> pl.DataFrame:
        return raw_observations

    def prepare_observations_in_realization(self, observations: pl.DataFrame, realization: int) -> pl.DataFrame:
        return observations

    def prepare_raw_responses(self, raw_responses: pl.LazyFrame, observations: pl.DataFrame) -> pl.LazyFrame:
        return raw_responses

    def prepare_pivoted_responses(self, pivoted_responses: pl.DataFrame, observations: pl.DataFrame, realization: int) -> pl.DataFrame:
        return pivoted_responses

    def match(self, observations: pl.DataFrame, pivoted: pl.DataFrame) -> pl.DataFrame:
        return observations.join(
            pivoted,
            how="left",
            on=["response_key", *self.response_cls.match_key],
            nulls_equal=True,
        )


class GenDataStrategy(ReservoirDataMatchStrategy[GenDataConfig]):
    pass


class SummaryStrategy(ReservoirDataMatchStrategy[SummaryConfig]):
    def match(self, observations: pl.DataFrame, pivoted: pl.DataFrame) -> pl.DataFrame:
        by_cols = [
            "response_key",
            *[k for k in self.response_cls.match_key if k != "time"],
        ]
        return observations.sort(
            by=[*by_cols, "time"]
        ).join_asof(
            pivoted.sort(by=[*by_cols, "time"]),
            by=by_cols,
            on="time",
            check_sortedness=False,  # Ref: https://github.com/pola-rs/polars/issues/21693
            strategy="nearest",
            tolerance="1s",
        )


class BreakthroughStrategy(ReservoirDataMatchStrategy[BreakthroughConfig]):
    def __init__(self, ensemble: LocalEnsemble, response_cls: BreakthroughConfig) -> None:
        self.ensemble_end_date = ensemble._compute_ensemble_end_date()
        super().__init__(ensemble, response_cls)

    def prepare_raw_responses(self, raw_responses: pl.LazyFrame, observations: pl.DataFrame) -> pl.LazyFrame:
        return self.response_cls.fill_missing_values(
            raw_responses, self.ensemble_end_date
        )


class RFTStrategy(ReservoirDataMatchStrategy[RFTConfig]):
    def prepare_observations_in_realization(self, observations: pl.DataFrame, realization: int) -> pl.DataFrame:
        observation_metadata_in_realization = self.ensemble.load_observation_location_metadata(
            realization
        )
        enriched_observations = RFTConfig.enrich_observations_with_metadata(
            observations, observation_metadata_in_realization
        )

        return qc_rft_observations(enriched_observations)

    def prepare_raw_responses(self, raw_responses: pl.LazyFrame, observations: pl.DataFrame) -> pl.LazyFrame:
        if (
            cast(
                RFTConfig, self.ensemble.experiment.response_configuration["rft"]
            ).approximate_missing_values
        ):
            raw_responses = RFTConfig.approximate_missing_rft_responses(
                raw_responses, observations
            )
        return raw_responses


class SeismicStrategy(ReservoirDataMatchStrategy[SeismicConfig]):
    def prepare_raw_observations(self, raw_observations: pl.DataFrame) -> pl.DataFrame:
        return qc_seismic_observations(
            raw_observations, self.ensemble.experiment.shape_registry
        )

    def prepare_pivoted_responses(self, pivoted_responses: pl.DataFrame, observations: pl.DataFrame, realization: int) -> pl.DataFrame:
        return (
            SeismicData.use_observation_locations_in_respective_responses(
                pivoted_responses, observations
            )
            # Due to performed validations, all match keys should be unique.
            # Calculating 'mean' anyway to assure no ugly error is raised if
            # validations are somehow bypassed
            .group_by(["response_key", "east", "north"])
            .agg(pl.col(str(realization)).mean())
        )


StrategyType = GenDataStrategy | SummaryStrategy | BreakthroughStrategy | RFTStrategy | SeismicStrategy


# @overload
# def matching_strategy(key: Literal["gen_data"], ensemble: LocalEnsemble, response_cls: GenDataConfig) -> GenDataStrategy: ...

# @overload
# def matching_strategy(key: Literal["summary"], ensemble: LocalEnsemble, response_cls: SummaryConfig) -> SummaryStrategy: ...

# @overload
# def matching_strategy(key: Literal["breakthrough"], ensemble: LocalEnsemble, response_cls: BreakthroughConfig) -> BreakthroughStrategy: ...

# @overload
# def matching_strategy(key: Literal["rft"], ensemble: LocalEnsemble, response_cls: RFTConfig) -> RFTStrategy: ...

# @overload
# def matching_strategy(key: Literal["seismic"], ensemble: LocalEnsemble, response_cls: SeismicConfig) -> SeismicStrategy: ...

# @overload
# def matching_strategy(key: str, ensemble: LocalEnsemble, response_cls: ResponseConfig) -> StrategyType: ...


def matching_strategy(key: str, ensemble: LocalEnsemble, response_cls: ResponseConfig) -> StrategyType:
    """Get reservoir strategy by string key."""
    data_map: dict[str, tuple[type, type]] = {
        "gen_data": (GenDataStrategy, ResponseConfig),
        "summary": (SummaryStrategy, SummaryConfig),
        "breakthrough": (BreakthroughStrategy, BreakthroughConfig),
        "rft": (RFTStrategy, RFTConfig),
        "seismic": (SeismicStrategy, SeismicConfig),
    }

    if key not in data_map:
        raise ValueError(f"Unknown reservoir data key: {key}")

    strategy_cls, expected_config_cls = data_map[key]
    if not isinstance(response_cls, expected_config_cls):
        raise TypeError(f"Expected {expected_config_cls.__name__}, got {type(response_cls).__name__}")

    return strategy_cls(ensemble, response_cls)


def validate_selected_observations(ensemble: LocalEnsemble, selected_observations: Iterable[str]) -> None:
    known_observations = ensemble.experiment.observation_keys
    unknown_observations = set(selected_observations) - set(known_observations)
    if unknown_observations:
        raise KeyError(
            f"Observations: {', '.join(unknown_observations)} not in experiment"
        )


def extract_observations(response_type: str, observations_by_type: dict[str, pl.DataFrame], selected_observations: Iterable[str]) -> pl.DataFrame:
    return (
        observations_by_type[response_type]
        .filter(pl.col("observation_key").is_in(list(selected_observations)))
        .with_columns([pl.col("response_key").cast(pl.Categorical)])
    )


def extract_responses(ensemble: LocalEnsemble, response_type: str, real: int) -> pl.LazyFrame:
    return ensemble._load_responses_lazy(
        response_type, (real,)
    ).with_columns([pl.col("response_key").cast(pl.Categorical)])


def filter_out_incorrectly_structured_responses(responses: pl.LazyFrame, observations: pl.DataFrame, response_cls: ResponseConfig) -> pl.LazyFrame:
    observed_cols = {
        k: observations[k].unique()
        for k in ["response_key", *response_cls.match_key]
    }

    # Filter out responses without observations
    for col, observed_values in observed_cols.items():
        if col not in {"time", "east", "north"}:
            responses = responses.filter(
                pl.col(col).is_in(observed_values.implode(), nulls_equal=True)
            )
    return responses


def update_error_on_mismatch(joined: pl.DataFrame, response_cls: ResponseConfig, real: str) -> pl.DataFrame:
    no_matched_response_condition = pl.col(str(real)).is_null()
    no_matched_response_error = pl.concat_str(
        [
            pl.lit("no response matched observation data: "),
            pl.lit("response_key="),
            pl.col("response_key").cast(pl.String),
            pl.lit(", "),
            response_cls.match_key_dict_expr(),
        ],
        separator="",
    )

    return joined.with_columns(
        append_to_qc_error(
            no_matched_response_condition, no_matched_response_error
        ).alias(f"qc_error_{real}")
    )


def add_index_column(joined: pl.DataFrame, response_cls: ResponseConfig, real: str) -> pl.DataFrame:
    # Avoid potential collision with "index" column (it could be a part
    # of the index_key)
    joined = joined.with_columns(
        response_cls.index_column_expr().alias("__tmp_index_key__")
    )
    if "index" in joined.columns:
        joined = joined.drop("index")
    joined = joined.rename({"__tmp_index_key__": "index"})

    return joined.with_columns(
        pl.when(pl.col(f"qc_error_{real}").is_null())
        .then(pl.col(str(real)))
        .otherwise(pl.lit(None, dtype=pl.Float32))
        .alias(str(real))
    )


def update_realization_column_on_error(joined: pl.DataFrame, real: str) -> pl.DataFrame:
    return joined.with_columns(
                pl.when(pl.col(f"qc_error_{real}").is_null())
                .then(pl.col(str(real)))
                .otherwise(pl.lit(None, dtype=pl.Float32))
                .alias(str(real))
            )


def retrieve_common_columns(joined: pl.DataFrame) -> pl.DataFrame:
    # The "leftmost" index columns are not yet collected.
    # They are the same for all iterations, and indexed the same
    # because we do a left join for the observations.
    # Hence, we select these columns only once.
    return joined.select(
        [
            "response_key",
            "index",
            "observation_key",
            "observations",
            "std",
            "east",
            "north",
            "radius",
        ]
    )


def prepare_raw_observations(strategy: StrategyType, observations_dict: dict[str, pl.DataFrame], selected_observations: Iterable[str]) -> pl.DataFrame:
    raw_observations = extract_observations(strategy.response_cls.type, observations_dict, selected_observations)
    return strategy.prepare_raw_observations(raw_observations)


def prepare_observations(raw_observations: pl.DataFrame, strategy: StrategyType, real: int) -> pl.DataFrame:
    observations = ensure_qc_error_column(raw_observations)
    return strategy.prepare_observations_in_realization(observations, real)


def prepare_raw_responses(ensemble: LocalEnsemble, strategy: StrategyType, real: int, observations: pl.DataFrame) -> pl.LazyFrame:
    response_type = strategy.response_cls.type
    responses = extract_responses(ensemble, response_type, real)
    responses = strategy.prepare_raw_responses(responses, observations)
    return filter_out_incorrectly_structured_responses(responses, observations, ensemble.experiment.response_configuration[response_type])


def prepare_pivoted_responses(raw_responses: pl.LazyFrame, strategy: StrategyType, observations: pl.DataFrame, real: int) -> pl.DataFrame:
    pivoted = raw_responses.collect(engine="streaming").pivot(
        on="realization",
        index=["response_key", *strategy.response_cls.match_key],
        values="values",
        aggregate_function="mean",
    )
    return strategy.prepare_pivoted_responses(pivoted, observations, real)


def prepare_responses(ensemble: LocalEnsemble, strategy: StrategyType, real: int, observations: pl.DataFrame) -> pl.DataFrame:
    raw_responses = prepare_raw_responses(ensemble, strategy, real, observations)
    return prepare_pivoted_responses(raw_responses, strategy, observations, real)


def join_responses_and_observations(observations: pl.DataFrame, responses: pl.DataFrame, strategy: StrategyType, real: int) -> pl.DataFrame:
    if responses.is_empty():
        # There are no responses for this realization,
        # so we explicitly create a column of Nones
        # to represent this. We are basically saying that
        # for this realization, each observation points
        # to a None response.
        joined = observations.with_columns(
            pl.lit(None, dtype=pl.Float32).alias(str(real)),
        )
    else:
        joined = strategy.match(observations, responses)

    joined = update_error_on_mismatch(joined, strategy.response_cls, str(real))
    joined = add_index_column(joined, strategy.response_cls, str(real))

    return update_realization_column_on_error(joined, str(real))


def match_observations_and_responses(
    ensemble: LocalEnsemble,
    selected_observations: Iterable[str],
    iens_active_index: npt.NDArray[np.int_],
) -> pl.DataFrame:
    """Fetches and aligns selected observations with their
    corresponding simulated responses from an ensemble.

    The returned DataFrame includes an "index" column containing a
    comma-separated string of the response type's index key values.
    Missing components are rendered as the literal "None" so that
    positional meaning is preserved:
    - Summary: "2024-01-15 00:00:00" (time)
    - GenData: "0, 42" (report_step, index)
    - RFT: "123.5, 456.7, 2500.0, ZONE_A" (east, north, tvd, zone)
            or "123.5, 456.7, 2500.0, None" when zone is missing
    - Seismic "123.5, 456.7" (east, north)
    """
    validate_selected_observations(ensemble, selected_observations)

    observations_dict = ensemble.experiment.observations
    response_config_dict = ensemble.experiment.response_configuration
    reals = np.sort(iens_active_index).tolist()

    dfs_per_response_type = []

    for (response_type, response_cls) in response_config_dict.items():
        if response_type not in observations_dict:
            continue
        strategy = matching_strategy(response_type, ensemble, response_cls)
        raw_observations = prepare_raw_observations(strategy, observations_dict, selected_observations)

        # Load and join one realization at a time to reduce peak memory usage
        common_columns: pl.DataFrame | None = None
        realization_columns: list[pl.DataFrame] = []

        for real in reals:
            observations = prepare_observations(raw_observations, strategy, real)
            responses = prepare_responses(ensemble, strategy, real, observations)
            joined = join_responses_and_observations(observations, responses, strategy, real)

            if common_columns is None:
                common_columns = retrieve_common_columns(joined)
            realization_columns.append(joined.select(str(real), f"qc_error_{real}"))

        if common_columns is None:
            # Not a single realization had any responses to the
            # observations. Hence, there is no need to include
            # it in the dataset
            continue

        dfs_per_response_type.append(
            pl.concat(
                [common_columns, *realization_columns],
                how="horizontal",
                strict=True,
            )
        )

    return pl.concat(dfs_per_response_type, how="vertical").with_columns(
        pl.col("response_key").cast(pl.String).alias("response_key")
    )


def match_rft_observations_and_responses_for_export(
    ensemble: LocalEnsemble,
) -> pl.DataFrame:
    """Fetches and aligns RFT observations with their corresponding
    simulated responses from an ensemble.

    Returns a DataFrame with observation/response data using
    column names equal to the ones used by the subscript forward model
    MERGE_RFT_ERTOBS, and compatible with the webviz-subsurface RftPlotter.
    """
    rft_observations = ensemble.experiment.observations.get("rft")
    if rft_observations is None or rft_observations.is_empty():
        raise ValueError("No RFT observations found in experiment")

    if "rft" not in ensemble.experiment.response_configuration:
        raise ValueError("No RFT response configuration found in experiment")

    realizations = ensemble.get_realization_list_with_responses()
    if not realizations:
        raise ValueError("No realizations with responses found")

    # Build date-to-report_step mapping from summary responses if available
    date_to_report_step: dict[str, int] = {}
    if "summary" in ensemble.experiment.response_configuration:
        try:
            summary_df = ensemble.load_responses("summary", (realizations[0],))
            times = summary_df["time"].unique().sort()
            for report_step, time in enumerate(times):
                date_str = time.strftime("%Y-%m-%d")
                date_to_report_step[date_str] = report_step
        except (KeyError, IndexError):
            pass  # No summary data available, will use default

    pure_observations = rft_observations.with_columns(
        pl.int_range(pl.len()).over("well").alias("order"),
    )

    pivot_index = [
        "well",
        "date",
        "realization",
        "well_connection_cell",
    ]

    output_columns = [
        "order",
        "east",
        "north",
        "md",
        "tvd",
        "zone",
        "pressure",
        "swat",
        "sgas",
        "soil",
        "valid_zone",
        "is_active",
        "i",
        "j",
        "k",
        "well",
        "date",
        "realization",
        "report_step",
        "observations",
        "std",
    ]

    result_frames: list[pl.DataFrame] = []

    for real in sorted(realizations):
        responses = ensemble.load_responses("rft", (real,)).with_columns(
            pl.col("property").str.to_lowercase(),
        )
        observation_metadata_in_realization = (
            ensemble.load_observation_location_metadata(real)
        )
        observations = RFTConfig.enrich_observations_with_metadata(
            pure_observations,
            observation_metadata_in_realization,
        )
        if cast(
            RFTConfig, ensemble.experiment.response_configuration["rft"]
        ).approximate_missing_values:
            responses = RFTConfig.approximate_missing_rft_responses(
                responses.lazy(), observations
            ).collect()

        join_keys = ["well", "date", "well_connection_cell"]
        observed_values = {k: observations[k].unique() for k in join_keys}

        for col, values in observed_values.items():
            responses = responses.filter(
                pl.col(col).is_in(values.implode(), nulls_equal=True)
            )

        pivoted = responses.pivot(
            on="property",
            index=pivot_index,
            values="values",
        )

        for col in ["pressure", "sgas", "swat"]:
            if col not in pivoted.columns:
                pivoted = pivoted.with_columns(pl.lit(None).cast(pl.Float32).alias(col))

        pivoted = pivoted.with_columns(
            pl.col("pressure").is_not_null().alias("is_active")
        )

        joined = (
            observations.join(
                pivoted,
                how="left",
                on=join_keys,
                nulls_equal=True,
            )
            .with_columns(
                RFTConfig.is_zone_valid().alias("valid_zone"),
                (1 - pl.col("sgas") - pl.col("swat")).alias("soil"),
                pl.col("is_active").fill_null(False),
                pl.col("date")
                .replace_strict(date_to_report_step, default=0)
                .alias("report_step"),
                pl.col("well_connection_cell").arr.get(0).alias("i"),
                pl.col("well_connection_cell").arr.get(1).alias("j"),
                pl.col("well_connection_cell").arr.get(2).alias("k"),
            )
            .select(output_columns)
        )

        result_frames.append(joined)

    return pl.concat(result_frames, how="vertical").rename(
        {
            "east": "utm_x",
            "north": "utm_y",
            "md": "measured_depth",
            "tvd": "true_vertical_depth",
            "date": "time",
            "observations": "observed",
            "std": "error",
        }
    )
