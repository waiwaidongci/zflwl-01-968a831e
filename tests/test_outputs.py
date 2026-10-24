from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest


ENTRY = os.environ.get("FEEDER_SOLVER", "/app/predict.py")

SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
PHASES = ("A", "B", "C")


def _run(payload: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-S", "-P", "-B", ENTRY, str(payload), str(output)],
        text=True,
        capture_output=True,
        env=SAFE_ENV,
        timeout=90,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _span_length(parent_index: int, child_index: int) -> float:
    return round(4.0 + ((parent_index * 13 + child_index * 7) % 37) / 2.0, 1)


def _network_case(
    profiles: list[list[float]],
    groups: list[list[int]],
    copper_coefficients: list[float],
    phase_members: list[list[int]],
    parents: list[list[int]],
    line_coefficient: float,
    tolerance: float = 0.000001,
    energy_tolerance: float = 0.000001,
) -> tuple[dict[str, object], dict[str, object]]:
    periods = len(profiles[0])
    meter_ids = [f"M{index + 1:03d}" for index in range(len(profiles))]
    transformer_ids = [f"T{index + 1:02d}" for index in range(len(groups))]
    root_labels = {index: phase for index, phase in enumerate(PHASES)}
    node_label = {
        index: transformer_ids[index] for index in range(len(transformer_ids))
    }
    loads = [
        [
            math.fsum(profiles[member][period] for member in members)
            for period in range(periods)
        ]
        for members in groups
    ]
    inlets = [
        [
            loads[index][period]
            + copper_coefficients[index] * loads[index][period] ** 2
            for period in range(periods)
        ]
        for index in range(len(groups))
    ]
    channels = {phase: [0.0] * periods for phase in PHASES}
    losses = {phase: [0.0] * periods for phase in PHASES}
    connections = []
    transformer_phase = {}
    for phase_index, (members, parent_array) in enumerate(
        zip(phase_members, parents)
    ):
        phase = PHASES[phase_index]
        position = {member: index for index, member in enumerate(members)}
        children = {member: [] for member in members}
        for node, parent in zip(members, parent_array):
            transformer_phase[node_label[node]] = phase
            parent_label = phase if parent == -1 else node_label[parent]
            connections.append(
                {"parent": parent_label, "child": node_label[node]}
            )
            if parent != -1:
                children[parent].append(node)
            for period in range(periods):
                losses[phase][period] += (
                    copper_coefficients[node] * loads[node][period] ** 2
                )

        def collect(node: int) -> tuple[float, ...]:
            carry = list(inlets[node])
            for child in children[node]:
                child_carry = collect(child)
                for period in range(periods):
                    carry[period] += child_carry[period]
            parent = parent_array[position[node]]
            parent_label = phase if parent == -1 else node_label[parent]
            length = _span_length(
                phase_index if parent == -1 else parent + 3,
                node + 3,
            )
            for period in range(periods):
                line_loss = line_coefficient * length * carry[period] ** 2
                losses[phase][period] += line_loss
                channels[phase][period] += line_loss
                channels[phase][period] += inlets[node][period]
            return tuple(carry)

        collect(members[0])

    spans = []
    for parent_index, parent in enumerate((*PHASES, *transformer_ids)):
        for child_index, child in enumerate(transformer_ids):
            if parent == child:
                continue
            spans.append(
                {
                    "parent": parent,
                    "child": child,
                    "length_m": _span_length(parent_index, child_index + 3),
                }
            )

    meter_to_transformer = {
        meter_ids[member]: transformer_ids[index]
        for index, members in enumerate(groups)
        for member in members
    }
    connections.sort(key=lambda item: item["child"])
    payload = {
        "schema": "feeder-tree-v3",
        "periods": periods,
        "tolerance_kw": tolerance,
        "energy_tolerance_kwh": energy_tolerance,
        "meters": [
            {"id": meter_ids[index], "profile": profiles[index]}
            for index in range(len(profiles))
        ],
        "transformers": [
            {
                "id": transformer_ids[index],
                "service_points": len(groups[index]),
                "energy_kwh": round(
                    0.25
                    * math.fsum(
                        math.fsum(profiles[member])
                        for member in groups[index]
                    ),
                    6,
                ),
                "copper_loss_kw_inv": copper_coefficients[index],
            }
            for index in range(len(groups))
        ],
        "spans": spans,
        "channels": channels,
    }
    expected = {
        "connections": connections,
        "line_loss_kw_inv": round(line_coefficient, 8),
        "losses_kw": {
            phase: [round(value, 6) for value in losses[phase]]
            for phase in PHASES
        },
        "meter_to_transformer": meter_to_transformer,
        "transformer_phase": transformer_phase,
    }
    return payload, expected


def _small_case(
    line_coefficient: float = 0.00008437,
    tolerance: float = 0.000001,
) -> tuple[dict[str, object], dict[str, object]]:
    return _network_case(
        profiles=[
            [12.0, 9.0, 15.0, 8.0, 10.0, 11.0],
            [7.0, 19.0, 11.0, 13.0, 9.0, 14.0],
            [20.0, 6.0, 14.0, 10.0, 17.0, 9.0],
            [8.0, 15.0, 7.0, 16.0, 12.0, 13.0],
            [18.0, 11.0, 19.0, 7.0, 15.0, 12.0],
            [9.0, 13.0, 10.0, 18.0, 8.0, 16.0],
        ],
        groups=[[0, 1], [2, 3], [4, 5]],
        copper_coefficients=[0.00005, 0.00007, 0.00004],
        phase_members=[[0], [1], [2]],
        parents=[[-1], [-1], [-1]],
        line_coefficient=line_coefficient,
        tolerance=tolerance,
    )


def _branch_case(
    line_coefficient: float = 0.00008437,
) -> tuple[dict[str, object], dict[str, object]]:
    return _network_case(
        profiles=[
            [9.0, 12.0, 8.0, 14.0, 11.0, 10.0],
            [7.0, 6.0, 11.0, 9.0, 8.0, 12.0],
            [14.0, 15.0, 13.0, 10.0, 16.0, 11.0],
            [18.0, 17.0, 19.0, 16.0, 20.0, 15.0],
            [5.0, 8.0, 6.0, 9.0, 7.0, 10.0],
            [21.0, 20.0, 22.0, 19.0, 23.0, 18.0],
            [11.0, 13.0, 12.0, 15.0, 14.0, 16.0],
            [19.0, 18.0, 20.0, 17.0, 21.0, 22.0],
        ],
        groups=[[0, 1], [2, 3], [4, 5], [6, 7]],
        copper_coefficients=[0.00005, 0.00007, 0.00004, 0.00006],
        phase_members=[[0, 1], [2], [3]],
        parents=[[-1, 0], [-1], [-1]],
        line_coefficient=line_coefficient,
    )


def _large_case() -> tuple[dict[str, object], dict[str, object]]:
    rng = random.Random(20261023)
    periods = 6
    profiles = [
        [round(rng.uniform(5, 95), 3) for _ in range(periods)]
        for _ in range(12)
    ]
    order = list(range(12))
    rng.shuffle(order)
    groups = [order[index * 2 : index * 2 + 2] for index in range(6)]
    copper = [round(rng.uniform(0.00003, 0.00015), 7) for _ in range(6)]
    return _network_case(
        profiles=profiles,
        groups=groups,
        copper_coefficients=copper,
        phase_members=[[0, 1, 2], [3, 4], [5]],
        parents=[[-1, 0, 0], [-1, 3], [-1]],
        line_coefficient=0.00008437,
    )


def test_branching_network_recovery_closes_all_phases(tmp_path: Path) -> None:
    """题面：「若客户表归属、变压器相别、定向父节点和共用线路系数只有一组满足全部关系」"""
    # case_id: NET-JOINT NET-METERS-MAX NET-TRANSFORMERS-MAX NET-PERIODS-MAX
    # case_id: NET-ASSIGN NET-SERVICE-COUNT NET-SINGLE-PHASE NET-NONEMPTY-PHASE
    # case_id: NET-LINE-LOSS NET-LOSS-DEFINITION NET-UNIQUE
    # case_id: NET-OUTPUT-FIELDS NET-METER-MAP NET-TRANS-PHASE NET-CONNECTIONS
    # case_id: NET-CONNECTION-FIELDS NET-CONN-ORDER NET-LOSSES NET-LOSSES-PERIODS
    # case_id: NET-LINE-OUT NET-UNIQUE-OUTPUT NET-LINE-FIELD NET-TRANSFO-KEYS
    # case_id: NET-SPAN-KEYS NET-SPAN-ENDPOINTS NET-SPAN-FIELDS NET-SPAN-PARENT
    # case_id: NET-SPAN-CHILD NET-PARENT-RULE NET-ROOT-PATH NET-LINE-SUM
    # case_id: NET-UNIQUE-MAP NET-OUT-OBJ NET-OUT-OTHER NET-LOSS-PERIOD NET-LOSS-KEYS
    payload, expected = _large_case()
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert set(result) == {
        "connections",
        "line_loss_kw_inv",
        "losses_kw",
        "meter_to_transformer",
        "status",
        "transformer_phase",
    }
    assert result["status"] == "ok"
    assert result["meter_to_transformer"] == expected["meter_to_transformer"]
    assert result["transformer_phase"] == expected["transformer_phase"]
    assert result["connections"] == expected["connections"]
    assert result["line_loss_kw_inv"] == pytest.approx(
        expected["line_loss_kw_inv"], abs=0.00000001
    )
    assert result["losses_kw"] == pytest.approx(
        expected["losses_kw"], abs=0.000001
    )


def test_branch_and_leaf_share_the_same_parent(tmp_path: Path) -> None:
    """题面：「不同变压器可以接到同一个上游节点，因此一相可以是分支树，而不是单链。」"""
    # case_id: NET-BRANCH-SHAPE NET-BRANCH
    payload, expected = _branch_case()
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["connections"] == expected["connections"]
    assert result["transformer_phase"] == expected["transformer_phase"]


def test_three_rooted_trees_are_recovered(tmp_path: Path) -> None:
    """题面：「相根，不能形成环。」"""
    # case_id: NET-SMALL NET-METERS-MIN NET-TRANSFORMERS-MIN
    # case_id: NET-TOLERANCE-LOW NET-COPPER-LOSS NET-INLET NET-MEASUREMENT-TOL
    # case_id: NET-ENERGY-TOL-LOW NET-SMALL-ACYCLIC
    payload, expected = _small_case()
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["connections"] == expected["connections"]
    assert result["losses_kw"] == pytest.approx(
        expected["losses_kw"], abs=0.000001
    )


def test_fifteen_minute_profiles_use_quarter_hour_energy(tmp_path: Path) -> None:
    """题面：「15 分钟折合 0.25 小时」"""
    # case_id: NET-ENERGY-QUARTER-01 NET-ENERGY-WINDOW-01
    payload, expected = _small_case()
    for transformer, members in zip(
        payload["transformers"], [[0, 1], [2, 3], [4, 5]]
    ):
        expected_kwh = 0.25 * math.fsum(
            math.fsum(payload["meters"][member]["profile"]) for member in members
        )
        assert expected_kwh > 0.0
        assert transformer["energy_kwh"] == pytest.approx(
            expected_kwh, abs=0.000001
        )
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["meter_to_transformer"] == expected["meter_to_transformer"]
    assert result["connections"] == expected["connections"]


def test_ambiguous_forest_is_reported(tmp_path: Path) -> None:
    """题面：「若存在两组或更多不同网络，输出对象有且只有 `status` 和 `candidates` 两个字段。」"""
    # case_id: NET-AMBIG NET-AMBIG-CAP
    payload, _expected = _network_case(
        profiles=[[10.0, 10.0, 10.0, 10.0, 10.0, 10.0] for _ in range(6)],
        groups=[[0, 1], [2, 3], [4, 5]],
        copper_coefficients=[0.00005, 0.00005, 0.00005],
        phase_members=[[0], [1], [2]],
        parents=[[-1], [-1], [-1]],
        line_coefficient=0.00008437,
    )
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert set(result) == {"status", "candidates"}
    assert result["status"] == "ambiguous"
    assert result["candidates"] == 2


def test_maximum_shared_loss_coefficient_is_accepted(tmp_path: Path) -> None:
    """题面：「`g` 是 0.0000001 到 0.001 的实数」"""
    # case_id: NET-LINE-HIGH NET-SHARED-LOSS
    payload, expected = _small_case(line_coefficient=0.001)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["line_loss_kw_inv"] == pytest.approx(0.001, abs=0.00000001)
    assert result["connections"] == expected["connections"]


def test_minimum_shared_loss_coefficient_is_accepted(tmp_path: Path) -> None:
    """题面：「`g` 是 0.0000001 到 0.001 的实数」"""
    # case_id: NET-LINE-LOW
    payload, expected = _small_case(line_coefficient=0.0000001)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["line_loss_kw_inv"] == pytest.approx(0.0000001, abs=0.00000001)
    assert result["connections"] == expected["connections"]


def test_shared_loss_coefficient_over_limit_is_refused(tmp_path: Path) -> None:
    """题面：「`g` 是 0.0000001 到 0.001 的实数」"""
    # case_id: NET-LINE-HIGH-FAIL
    payload, _expected = _small_case(line_coefficient=0.001001)
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_period_minimum_is_accepted(tmp_path: Path) -> None:
    """题面：「`periods` 是 1 到 6 的整数。」"""
    # case_id: NET-PERIOD-LOW
    payload, expected = _network_case(
        profiles=[[12.0], [7.0], [20.0], [8.0], [18.0], [9.0]],
        groups=[[0, 1], [2, 3], [4, 5]],
        copper_coefficients=[0.00005, 0.00007, 0.00004],
        phase_members=[[0], [1], [2]],
        parents=[[-1], [-1], [-1]],
        line_coefficient=0.00008437,
    )
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["connections"] == expected["connections"]


def test_period_over_limit_is_refused(tmp_path: Path) -> None:
    """题面：「`periods` 是 1 到 6 的整数。」"""
    # case_id: NET-PERIOD-HIGH
    payload, _expected = _small_case()
    payload["periods"] = 7
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_meter_over_limit_is_refused(tmp_path: Path) -> None:
    """题面：「`meters` 有 6 到 12 个表对象」"""
    # case_id: NET-METERS-HIGH
    payload, _expected = _small_case()
    payload["meters"].append({"id": "M013", "profile": [1.0] * 6})
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_transformer_over_limit_is_refused(tmp_path: Path) -> None:
    """题面：「`transformers` 有 3 到 6 台设备」"""
    # case_id: NET-TRANSFORMERS-HIGH
    payload, _expected = _small_case()
    payload["transformers"].append(
        {
            "id": "T07",
            "service_points": 1,
            "energy_kwh": 10.0,
            "copper_loss_kw_inv": 0.00005,
        }
    )
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_invalid_transformer_id_is_refused(tmp_path: Path) -> None:
    """题面：「编号写成 `T01` 这种两位数字形式」"""
    # case_id: NET-TRANSFORMER-ID
    payload, _expected = _small_case()
    payload["transformers"][0]["id"] = "T1"
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_service_point_one_is_accepted(tmp_path: Path) -> None:
    """题面：「`service_points` 是 1 到 2 的整数」"""
    # case_id: NET-SERVICE-LOW
    payload, expected = _network_case(
        profiles=[
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            [20.0, 19.0, 18.0, 17.0, 16.0, 15.0],
            [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ],
        groups=[[0], [1, 2], [3, 4], [5, 6]],
        copper_coefficients=[0.00005, 0.00007, 0.00004, 0.00006],
        phase_members=[[0, 1], [2], [3]],
        parents=[[-1, 0], [-1], [-1]],
        line_coefficient=0.00008437,
    )
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["connections"] == expected["connections"]


def test_service_point_over_limit_is_refused(tmp_path: Path) -> None:
    """题面：「`service_points` 是 1 到 2 的整数」"""
    # case_id: NET-SERVICE-HIGH-FAIL
    payload, _expected = _small_case()
    payload["transformers"][0]["service_points"] = 3
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_energy_tolerance_over_limit_is_refused(tmp_path: Path) -> None:
    """题面：「`energy_tolerance_kwh` 是 0 到 0.01」"""
    # case_id: NET-ENERGY-TOL-HIGH
    payload, _expected = _small_case()
    payload["energy_tolerance_kwh"] = 0.0101
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_missing_span_is_refused(tmp_path: Path) -> None:
    """题面：「三个相根到每台变压器、以及任意两台变压器之间的定向电缆段都要在 `spans` 中出现。」"""
    # case_id: NET-SPAN-MISSING
    payload, _expected = _small_case()
    payload["spans"].pop()
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_duplicate_span_is_refused(tmp_path: Path) -> None:
    """题面：「同一父节点到同一子节点不得重复」"""
    # case_id: NET-SPAN-DUP
    payload, _expected = _small_case()
    payload["spans"][1] = dict(payload["spans"][0])
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_span_length_over_limit_is_refused(tmp_path: Path) -> None:
    """题面：「长度是 0.1 到 1000 的有限实数。」"""
    # case_id: NET-SPAN-LENGTH
    payload, _expected = _small_case()
    payload["spans"][0]["length_m"] = 1000.1
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_duplicate_meter_id_is_refused(tmp_path: Path) -> None:
    """题面：「表号写成 `M001` 这种三位数字形式，同一输入内不得重号。」"""
    # case_id: NET-DUP
    payload, _expected = _small_case()
    payload["meters"][1]["id"] = payload["meters"][0]["id"]
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_invalid_meter_id_is_refused(tmp_path: Path) -> None:
    """题面：「表号写成 `M001` 这种三位数字形式」"""
    # case_id: NET-METER-ID
    payload, _expected = _small_case()
    payload["meters"][0]["id"] = "M01"
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_profile_wrong_length_is_refused(tmp_path: Path) -> None:
    """题面：「`profile` 有 `periods` 个非负」"""
    # case_id: NET-PROFILE-LENGTH
    payload, _expected = _small_case()
    payload["meters"][0]["profile"].pop()
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_boolean_meter_reading_is_refused(tmp_path: Path) -> None:
    """题面：「`profile` 有 `periods` 个非负」"""
    # case_id: NET-BOOL
    payload, _expected = _small_case()
    payload["meters"][0]["profile"][0] = True
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_negative_channel_reading_is_refused(tmp_path: Path) -> None:
    """题面：「表示馈线首端对应相在一个时段内的有功功率」"""
    # case_id: NET-CHANNEL-NEGATIVE
    payload, _expected = _small_case()
    payload["channels"]["A"][0] = -0.5
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_tolerance_over_limit_is_refused(tmp_path: Path) -> None:
    """题面：「功率容差写成 `tolerance_kw`，数值限制在 0 到 0.005 的闭区间内」"""
    # case_id: NET-TOLERANCE-HIGH-FAIL
    payload, _expected = _small_case()
    payload["tolerance_kw"] = 0.0051
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_extra_top_level_key_is_refused(tmp_path: Path) -> None:
    """题面：「输入外层对象有且只有」"""
    # case_id: NET-ROOT NET-ROOT-EXTRA NET-ROOT-SCHEMA NET-ROOT-NETWORK
    payload, _expected = _small_case()
    payload["notes"] = "unexpected"
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_extra_phase_channel_is_refused(tmp_path: Path) -> None:
    """题面：「`channels` 有且只有 `A`、`B`、`C` 三个字段。」"""
    # case_id: NET-CHANNELS NET-CHANNEL-EXACT
    payload, _expected = _small_case()
    payload["channels"]["D"] = [0.0] * 6
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_wrong_schema_is_refused(tmp_path: Path) -> None:
    """题面：「`schema` 固定写成 `"feeder-tree-v3"`。」"""
    # case_id: NET-SCHEMA
    payload, _expected = _small_case()
    payload["schema"] = "feeder-net-v2"
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_duplicate_transformer_id_is_refused(tmp_path: Path) -> None:
    """题面：「编号写成 `T01` 这种两位数字形式，同一输入内不得重号。」"""
    # case_id: NET-TRANSFORMER-DUP
    payload, _expected = _small_case()
    payload["transformers"][1]["id"] = payload["transformers"][0]["id"]
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_no_feasible_tree_is_refused(tmp_path: Path) -> None:
    """题面：「本次复盘失败」"""
    # case_id: NET-NO-SOLUTION NET-REFUSE NET-EXIT-CODE
    payload, _expected = _small_case()
    payload["channels"] = {
        "A": [1000000.0] * 6,
        "B": [1000000.0] * 6,
        "C": [1000000.0] * 6,
    }
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("REFUSED: ")
    assert not output.exists()


def test_output_is_compact_sorted_and_newline_terminated(tmp_path: Path) -> None:
    """题面：「输出 JSON 使用紧凑格式，键按字典序排列，末尾带一个换行。」"""
    # case_id: NET-ORDER NET-NEWLINE
    payload, _expected = _small_case()
    source = tmp_path / "input.json"
    output = tmp_path / "result.json"
    _write_json(source, payload)
    completed = _run(source, output)
    raw = output.read_bytes()
    parsed = json.loads(raw)
    canonical = (
        json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    assert completed.returncode == 0
    assert raw == canonical


def test_repeated_run_has_identical_bytes(tmp_path: Path) -> None:
    """题面：「程序不读取测试目录」"""
    # case_id: NET-CLI NET-HARDCODE NET-STDLIB NET-NOTESTS NET-ENTRY
    # case_id: NET-REPLAY NET-REPLAY-BYTES
    payload, expected = _small_case()
    source = tmp_path / "input.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_json(source, payload)
    first_run = _run(source, first)
    second_run = _run(source, second)
    first_result = json.loads(first.read_text(encoding="utf-8"))
    assert first_run.returncode == 0
    assert second_run.returncode == 0
    assert first.read_bytes() == second.read_bytes()
    assert first_result["connections"] == expected["connections"]
