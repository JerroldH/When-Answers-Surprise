from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterator


@dataclass(frozen=True)
class PartitionConfig:
    type: str
    start: int | str
    end: int | str


@dataclass(frozen=True)
class QueryPartition:
    name: str
    start_date: date
    end_date: date

    @property
    def params(self) -> dict[str, date]:
        return {"start_date": self.start_date, "end_date": self.end_date}


def generate_partitions(config: PartitionConfig) -> list[QueryPartition]:
    if config.type == "year":
        return list(_year_partitions(config.start, config.end))
    if config.type == "month":
        return list(_month_partitions(config.start, config.end))
    raise ValueError("partition.type must be 'year' or 'month'.")


def _year_partitions(start: int | str, end: int | str) -> Iterator[QueryPartition]:
    start_year = _parse_year(start, "partition.start")
    end_year = _parse_year(end, "partition.end")
    if end_year < start_year:
        raise ValueError("partition.end must be greater than or equal to partition.start.")
    for year in range(start_year, end_year + 1):
        yield QueryPartition(
            name=f"year={year}",
            start_date=date(year, 1, 1),
            end_date=date(year + 1, 1, 1),
        )


def _month_partitions(start: int | str, end: int | str) -> Iterator[QueryPartition]:
    start_year, start_month = _parse_month(start, "partition.start")
    end_year, end_month = _parse_month(end, "partition.end")
    cursor_year, cursor_month = start_year, start_month
    if (end_year, end_month) < (start_year, start_month):
        raise ValueError("partition.end must be greater than or equal to partition.start.")
    while (cursor_year, cursor_month) <= (end_year, end_month):
        next_year = cursor_year + 1 if cursor_month == 12 else cursor_year
        next_month = 1 if cursor_month == 12 else cursor_month + 1
        yield QueryPartition(
            name=f"month={cursor_year:04d}-{cursor_month:02d}",
            start_date=date(cursor_year, cursor_month, 1),
            end_date=date(next_year, next_month, 1),
        )
        cursor_year, cursor_month = next_year, next_month


def _parse_year(value: int | str, label: str) -> int:
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a four-digit year.") from exc
    if year < 1900 or year > 2200:
        raise ValueError(f"{label} is outside the supported range 1900-2200.")
    return year


def _parse_month(value: int | str, label: str) -> tuple[int, int]:
    text = str(value)
    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if not match:
        raise ValueError(f"{label} must use YYYY-MM for monthly partitions.")
    year = _parse_year(match.group(1), label)
    month = int(match.group(2))
    if month < 1 or month > 12:
        raise ValueError(f"{label} month must be between 01 and 12.")
    return year, month
