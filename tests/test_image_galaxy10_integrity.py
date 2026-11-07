import ast
import hashlib
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "notebooks" / "image_galaxy10_eda.py"


def load_source_function(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def test_md5_matches_the_expected_file_content(tmp_path):
    data_file = tmp_path / "sample.h5"
    data_file.write_bytes(b"galaxy10")
    expected_hash = hashlib.md5(b"galaxy10").hexdigest()

    assert load_source_function("md5_matches")(data_file, expected_hash)


def test_md5_rejects_a_different_hash(tmp_path):
    data_file = tmp_path / "sample.h5"
    data_file.write_bytes(b"galaxy10")

    assert not load_source_function("md5_matches")(data_file, "0" * 32)


def test_select_model_uses_the_higher_validation_macro_f1():
    logistic_scores = {"macro_f1": 0.5952}
    transfer_scores = {"macro_f1": 0.4357}

    assert load_source_function("select_model")(logistic_scores, transfer_scores) == "Pretrained features + Logistic"


def test_select_model_can_choose_the_transfer_head():
    logistic_scores = {"macro_f1": 0.40}
    transfer_scores = {"macro_f1": 0.41}

    assert load_source_function("select_model")(logistic_scores, transfer_scores) == "Frozen MobileNetV3 head"
