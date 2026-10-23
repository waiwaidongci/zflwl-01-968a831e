#!/usr/bin/env python3
"""Reference solver for the branching feeder network inversion task."""

from __future__ import annotations

import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterator


INPUT_KEYS = {
    "schema",
    "periods",
    "tolerance_kw",
    "energy_tolerance_kwh",
    "meters",
    "transformers",
    "spans",
    "channels",
}
PHASES = ("A", "B", "C")
METER_ID = re.compile(r"M[0-9]{3}\Z")
TRANSFORMER_ID = re.compile(r"T[0-9]{2}\Z")


class Refused(ValueError):
    """Input or observations cannot produce a publishable network."""


def _number(value: Any, label: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise Refused(f"{label} is outside its allowed range")
    return number


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refused(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise Refused(f"{label} is outside its allowed range")
    return value


def parse_input(path: Path) -> tuple[
    int,
    float,
    float,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], float],
    dict[str, list[float]],
]:
    """Read and validate the complete feeder-tree-v3 input contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refused("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise Refused("input is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != INPUT_KEYS:
        raise Refused("top-level keys do not match feeder-tree-v3")
    if payload.get("schema") != "feeder-tree-v3":
        raise Refused("unsupported schema")

    periods = _integer(payload.get("periods"), "periods", 1, 6)
    tolerance = _number(payload.get("tolerance_kw"), "tolerance_kw", 0.0)
    if tolerance > 0.005:
        raise Refused("tolerance_kw exceeds the public limit")
    energy_tolerance = _number(
        payload.get("energy_tolerance_kwh"),
        "energy_tolerance_kwh",
        0.0,
    )
    if energy_tolerance > 0.01:
        raise Refused("energy_tolerance_kwh exceeds the public limit")

    raw_meters = payload.get("meters")
    if not isinstance(raw_meters, list) or not 6 <= len(raw_meters) <= 12:
        raise Refused("meters must contain between 6 and 12 entries")
    meters: list[dict[str, Any]] = []
    seen_meters: set[str] = set()
    for index, item in enumerate(raw_meters):
        if not isinstance(item, dict) or set(item) != {"id", "profile"}:
            raise Refused(f"meter {index} has an invalid key set")
        meter_id = item.get("id")
        if not isinstance(meter_id, str) or METER_ID.fullmatch(meter_id) is None:
            raise Refused(f"meter {index} has an invalid id")
        if meter_id in seen_meters:
            raise Refused("meter ids must be unique")
        seen_meters.add(meter_id)
        profile = item.get("profile")
        if not isinstance(profile, list) or len(profile) != periods:
            raise Refused(f"meter {meter_id} has an invalid profile length")
        meters.append(
            {
                "id": meter_id,
                "profile": [
                    _number(value, f"meter {meter_id} reading", 0.0)
                    for value in profile
                ],
            }
        )

    raw_transformers = payload.get("transformers")
    if (
        not isinstance(raw_transformers, list)
        or not 3 <= len(raw_transformers) <= 6
    ):
        raise Refused("transformers must contain between 3 and 6 entries")
    transformers: list[dict[str, Any]] = []
    seen_transformers: set[str] = set()
    service_total = 0
    for index, item in enumerate(raw_transformers):
        expected_keys = {
            "id",
            "service_points",
            "energy_kwh",
            "copper_loss_kw_inv",
        }
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise Refused(f"transformer {index} has an invalid key set")
        transformer_id = item.get("id")
        if (
            not isinstance(transformer_id, str)
            or TRANSFORMER_ID.fullmatch(transformer_id) is None
        ):
            raise Refused(f"transformer {index} has an invalid id")
        if transformer_id in seen_transformers:
            raise Refused("transformer ids must be unique")
        seen_transformers.add(transformer_id)
        service_points = _integer(
            item.get("service_points"),
            f"transformer {transformer_id} service_points",
            1,
            2,
        )
        service_total += service_points
        energy = _number(
            item.get("energy_kwh"),
            f"transformer {transformer_id} energy_kwh",
            0.0,
        )
        copper = _number(
            item.get("copper_loss_kw_inv"),
            f"transformer {transformer_id} copper_loss_kw_inv",
            0.0000001,
        )
        if copper > 0.01:
            raise Refused(
                f"transformer {transformer_id} copper_loss_kw_inv exceeds its limit"
            )
        transformers.append(
            {
                "id": transformer_id,
                "service_points": service_points,
                "energy_kwh": energy,
                "copper_loss_kw_inv": copper,
            }
        )
    if service_total != len(meters):
        raise Refused("service_points must sum to the meter count")

    transformer_ids = {item["id"] for item in transformers}
    expected_spans = {
        (phase, transformer_id)
        for phase in PHASES
        for transformer_id in transformer_ids
    }
    expected_spans.update(
        (parent, child)
        for parent in transformer_ids
        for child in transformer_ids
        if parent != child
    )
    raw_spans = payload.get("spans")
    if not isinstance(raw_spans, list) or len(raw_spans) != len(expected_spans):
        raise Refused("spans must contain every possible directed segment")
    spans: dict[tuple[str, str], float] = {}
    for index, item in enumerate(raw_spans):
        if not isinstance(item, dict) or set(item) != {
            "parent",
            "child",
            "length_m",
        }:
            raise Refused(f"span {index} has an invalid key set")
        parent = item.get("parent")
        child = item.get("child")
        if not isinstance(parent, str) or not isinstance(child, str):
            raise Refused(f"span {index} has invalid endpoints")
        pair = (parent, child)
        if pair not in expected_spans:
            raise Refused(f"span {index} is not an allowed connection")
        if pair in spans:
            raise Refused("span parent-child pairs must be unique")
        length = _number(item.get("length_m"), f"span {index} length", 0.1)
        if length > 1000.0:
            raise Refused(f"span {index} length exceeds its limit")
        spans[pair] = length
    if set(spans) != expected_spans:
        raise Refused("spans do not cover the complete candidate graph")

    raw_channels = payload.get("channels")
    if not isinstance(raw_channels, dict) or set(raw_channels) != set(PHASES):
        raise Refused("channels must contain exactly A, B and C")
    channels: dict[str, list[float]] = {}
    for phase in PHASES:
        values = raw_channels.get(phase)
        if not isinstance(values, list) or len(values) != periods:
            raise Refused(f"channel {phase} has an invalid length")
        channels[phase] = [
            _number(value, f"channel {phase} reading", 0.0) for value in values
        ]
    return (
        periods,
        tolerance,
        energy_tolerance,
        meters,
        transformers,
        spans,
        channels,
    )


def _groupings(
    position: int,
    order: tuple[int, ...],
    candidates: dict[int, tuple[tuple[int, ...], ...]],
    used: int,
    chosen: list[tuple[int, ...]],
) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Yield exact covers of meters by the transformer energy candidates."""
    if position == len(order):
        if used == (1 << len({meter for group in chosen for meter in group})) - 1:
            yield tuple(chosen)
        return
    transformer_index = order[position]
    for group in candidates[transformer_index]:
        mask = sum(1 << member for member in group)
        if used & mask:
            continue
        chosen.append(group)
        yield from _groupings(
            position + 1,
            order,
            candidates,
            used | mask,
            chosen,
        )
        chosen.pop()


def _energy_groupings(
    meters: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    energy_tolerance: float,
) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Yield all meter groupings compatible with transformer window energy."""
    energies = [
        0.25 * math.fsum(meter["profile"])
        for meter in meters
    ]
    candidates: dict[int, tuple[tuple[int, ...], ...]] = {}
    for transformer_index, transformer in enumerate(transformers):
        size = transformer["service_points"]
        options = []
        for members in itertools.combinations(range(len(meters)), size):
            total = math.fsum(energies[member] for member in members)
            if (
                abs(total - transformer["energy_kwh"])
                <= energy_tolerance + 1e-12
            ):
                options.append(members)
        candidates[transformer_index] = tuple(options)
        if not options:
            return
    order = tuple(
        sorted(candidates, key=lambda index: (len(candidates[index]), index))
    )
    meter_count = len(meters)
    full_mask = (1 << meter_count) - 1
    chosen: list[tuple[int, ...]] = []

    def visit(position: int, used: int) -> Iterator[tuple[tuple[int, ...], ...]]:
        if position == len(order):
            if used == full_mask:
                result = [()] * len(transformers)
                for order_position, transformer_index in enumerate(order):
                    result[transformer_index] = chosen[order_position]
                yield tuple(result)
            return
        transformer_index = order[position]
        for group in candidates[transformer_index]:
            mask = sum(1 << member for member in group)
            if used & mask:
                continue
            chosen.append(group)
            yield from visit(position + 1, used | mask)
            chosen.pop()

    yield from visit(0, 0)


def _phase_memberships(
    transformer_count: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return every assignment of all transformers onto three non-empty phases."""
    result = []
    for slots in itertools.product(range(3), repeat=transformer_count):
        groups = tuple(
            tuple(
                index
                for index, slot in enumerate(slots)
                if slot == phase_index
            )
            for phase_index in range(3)
        )
        if all(groups):
            result.append(groups)
    return tuple(result)


def _tree_parents(
    members: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate rooted tree parent arrays; -1 denotes the phase root."""
    options = []
    choices = (-1, *members)
    for parents in itertools.product(choices, repeat=len(members)):
        valid = True
        for node in members:
            seen = set()
            current = node
            while current != -1:
                if current in seen:
                    valid = False
                    break
                seen.add(current)
                current = parents[members.index(current)]
            if not valid:
                break
        if valid:
            options.append(parents)
    return tuple(options)


def _candidate_networks(
    periods: int,
    tolerance: float,
    energy_tolerance: float,
    meters: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    spans: dict[tuple[str, str], float],
    channels: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Return at most two rooted forests consistent with all observations."""
    transformer_ids = tuple(item["id"] for item in transformers)
    copper_coefficients = tuple(
        item["copper_loss_kw_inv"] for item in transformers
    )
    profiles = tuple(tuple(item["profile"]) for item in meters)
    root_total = tuple(
        math.fsum(channels[phase][period] for phase in PHASES)
        for period in range(periods)
    )
    phase_groups = _phase_memberships(len(transformers))
    tree_cache: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
    candidates: list[dict[str, Any]] = []

    for groups in _energy_groupings(meters, transformers, energy_tolerance):
        loads = [
            tuple(
                math.fsum(profiles[member][period] for member in group)
                for period in range(periods)
            )
            for group in groups
        ]
        inlets = tuple(
            tuple(
                loads[index][period]
                + copper_coefficients[index] * loads[index][period] ** 2
                for period in range(periods)
            )
            for index in range(len(transformers))
        )
        load_total = tuple(
            math.fsum(loads[index][period] for index in range(len(transformers)))
            for period in range(periods)
        )
        copper_total = tuple(
            math.fsum(
                copper_coefficients[index] * loads[index][period] ** 2
                for index in range(len(transformers))
            )
            for period in range(periods)
        )
        conservation_residual = tuple(
            root_total[period] - load_total[period] - copper_total[period]
            for period in range(periods)
        )
        subset_sums: list[tuple[float, ...]] = [
            (0.0,) * periods for _ in range(1 << len(transformers))
        ]
        for mask in range(1, 1 << len(transformers)):
            bit = mask & -mask
            index = bit.bit_length() - 1
            previous = subset_sums[mask ^ bit]
            subset_sums[mask] = tuple(
                previous[period] + inlets[index][period]
                for period in range(periods)
            )

        for memberships in phase_groups:
            masks = tuple(
                sum(1 << member for member in members)
                for members in memberships
            )
            feasible = True
            for phase_index, mask in enumerate(masks):
                for period in range(periods):
                    if (
                        subset_sums[mask][period]
                        > channels[PHASES[phase_index]][period] + tolerance + 1e-9
                    ):
                        feasible = False
                        break
                if not feasible:
                    break
            if not feasible:
                continue
            phase_trees = []
            for members in memberships:
                if members not in tree_cache:
                    tree_cache[members] = _tree_parents(members)
                phase_trees.append(tree_cache[members])

            for selected_parents in itertools.product(*phase_trees):
                weighted_square_total = [0.0] * periods
                valid_tree = True
                for phase_index, (members, parents) in enumerate(
                    zip(memberships, selected_parents)
                ):
                    position = {member: index for index, member in enumerate(members)}
                    children: dict[int, list[int]] = {member: [] for member in members}
                    for node, parent in zip(members, parents):
                        if parent != -1:
                            children[parent].append(node)

                    def subtree(node: int) -> tuple[float, ...]:
                        carry = list(inlets[node])
                        for child in children[node]:
                            child_carry = subtree(child)
                            for period in range(periods):
                                carry[period] += child_carry[period]
                        parent = parents[position[node]]
                        parent_label = PHASES[phase_index] if parent == -1 else transformer_ids[parent]
                        edge_key = (parent_label, transformer_ids[node])
                        if edge_key not in spans:
                            nonlocal_valid[0] = False
                            return tuple(carry)
                        length = spans[edge_key]
                        for period in range(periods):
                            weighted_square_total[period] += (
                                length * carry[period] ** 2
                            )
                        return tuple(carry)

                    nonlocal_valid = [True]
                    subtree(members[0])
                    if not nonlocal_valid[0]:
                        valid_tree = False
                        break
                if not valid_tree:
                    continue
                denominator = math.fsum(
                    value * value for value in weighted_square_total
                )
                if denominator <= 0.0:
                    continue
                coefficient = (
                    math.fsum(
                        conservation_residual[period]
                        * weighted_square_total[period]
                        for period in range(periods)
                    )
                    / denominator
                )
                if not 0.0000001 - 1e-12 <= coefficient <= 0.001 + 1e-12:
                    continue

                valid = True
                for phase_index, (members, parents) in enumerate(
                    zip(memberships, selected_parents)
                ):
                    position = {member: index for index, member in enumerate(members)}
                    children: dict[int, list[int]] = {member: [] for member in members}
                    for node, parent in zip(members, parents):
                        if parent != -1:
                            children[parent].append(node)
                    weighted_square = [0.0] * periods

                    def collect(node: int) -> tuple[float, ...]:
                        carry = list(inlets[node])
                        for child in children[node]:
                            child_carry = collect(child)
                            for period in range(periods):
                                carry[period] += child_carry[period]
                        parent = parents[position[node]]
                        parent_label = PHASES[phase_index] if parent == -1 else transformer_ids[parent]
                        length = spans[(parent_label, transformer_ids[node])]
                        for period in range(periods):
                            weighted_square[period] += length * carry[period] ** 2
                        return tuple(carry)

                    collect(members[0])
                    for period in range(periods):
                        predicted = (
                            subset_sums[masks[phase_index]][period]
                            + coefficient * weighted_square[period]
                        )
                        if (
                            abs(predicted - channels[PHASES[phase_index]][period])
                            > tolerance + 1e-9
                        ):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                candidate = {
                    "groups": groups,
                    "memberships": memberships,
                    "parents": selected_parents,
                    "coefficient": coefficient,
                }
                if not any(
                    existing["groups"] == candidate["groups"]
                    and existing["memberships"] == candidate["memberships"]
                    and existing["parents"] == candidate["parents"]
                    for existing in candidates
                ):
                    candidates.append(candidate)
                if len(candidates) == 2:
                    return candidates
    return candidates


def recover(
    periods: int,
    tolerance: float,
    energy_tolerance: float,
    meters: list[dict[str, Any]],
    transformers: list[dict[str, Any]],
    spans: dict[tuple[str, str], float],
    channels: dict[str, list[float]],
) -> dict[str, Any]:
    """Infer the unique branching feeder network or report a second one."""
    candidates = _candidate_networks(
        periods,
        tolerance,
        energy_tolerance,
        meters,
        transformers,
        spans,
        channels,
    )
    if not candidates:
        raise Refused("channels do not admit a feeder network")
    if len(candidates) > 1:
        return {"candidates": 2, "status": "ambiguous"}

    candidate = candidates[0]
    transformer_ids = tuple(item["id"] for item in transformers)
    meter_to_transformer: dict[str, str] = {}
    for transformer_index, members in enumerate(candidate["groups"]):
        for member in members:
            meter_to_transformer[meters[member]["id"]] = transformer_ids[
                transformer_index
            ]

    transformer_phase: dict[str, str] = {}
    connections: list[dict[str, str]] = []
    for phase, members, parents in zip(
        PHASES,
        candidate["memberships"],
        candidate["parents"],
    ):
        for node, parent in zip(members, parents):
            transformer_phase[transformer_ids[node]] = phase
            parent_label = phase if parent == -1 else transformer_ids[parent]
            connections.append(
                {
                    "parent": parent_label,
                    "child": transformer_ids[node],
                }
            )
    connections.sort(key=lambda item: item["child"])

    loads = [
        [
            math.fsum(meters[member]["profile"][period] for member in group)
            for period in range(periods)
        ]
        for group in candidate["groups"]
    ]
    inlets = [
        [
            loads[index][period]
            + transformers[index]["copper_loss_kw_inv"]
            * loads[index][period]
            ** 2
            for period in range(periods)
        ]
        for index in range(len(transformers))
    ]
    coefficient = candidate["coefficient"]
    losses: dict[str, list[float]] = {}
    for phase_index, (members, parents) in enumerate(
        zip(candidate["memberships"], candidate["parents"])
    ):
        position = {member: index for index, member in enumerate(members)}
        children: dict[int, list[int]] = {member: [] for member in members}
        for node, parent in zip(members, parents):
            if parent != -1:
                children[parent].append(node)
        phase_loss = [0.0] * periods
        for member in members:
            for period in range(periods):
                phase_loss[period] += (
                    transformers[member]["copper_loss_kw_inv"]
                    * loads[member][period]
                    ** 2
                )

        def collect(node: int) -> tuple[float, ...]:
            carry = list(inlets[node])
            for child in children[node]:
                child_carry = collect(child)
                for period in range(periods):
                    carry[period] += child_carry[period]
            parent = parents[position[node]]
            parent_label = PHASES[phase_index] if parent == -1 else transformer_ids[parent]
            length = spans[(parent_label, transformer_ids[node])]
            for period in range(periods):
                phase_loss[period] += coefficient * length * carry[period] ** 2
            return tuple(carry)

        collect(members[0])
        losses[PHASES[phase_index]] = [
            round(value, 6) for value in phase_loss
        ]

    return {
        "connections": connections,
        "line_loss_kw_inv": round(coefficient, 8),
        "losses_kw": losses,
        "meter_to_transformer": dict(sorted(meter_to_transformer.items())),
        "status": "ok",
        "transformer_phase": dict(sorted(transformer_phase.items())),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("REFUSED: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        (
            periods,
            tolerance,
            energy_tolerance,
            meters,
            transformers,
            spans,
            channels,
        ) = parse_input(source)
        result = recover(
            periods,
            tolerance,
            energy_tolerance,
            meters,
            transformers,
            spans,
            channels,
        )
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    target.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
