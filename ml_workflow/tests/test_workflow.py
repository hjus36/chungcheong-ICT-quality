"""ml_workflow 종단 테스트 — 작은 합성 데이터셋으로 전 단계를 빠르게 돌린다.

    python -m unittest discover -s tests -v      (ml_workflow 폴더에서)

실제 데이터(dataset_V2.parquet)는 쓰지 않는다. ICT_CONFIG_DIR·ICT_DATA_PATH·ICT_RUNS_DIR로
설정·데이터·실행 폴더를 tests/_tmp/ 아래 임시 위치로 돌린다.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

WF = Path(__file__).resolve().parents[1]
TMP = WF / "tests" / "_tmp"


def make_dataset(path: Path, seed: int = 7) -> None:
    """실험일 4 × 실험 6 × 사이클 10 = 240행. 불량은 보압이 낮을수록 늘어난다."""
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for d, day in enumerate(["2022-03-01", "2022-03-02", "2022-03-08", "2022-03-09"]):
        for e in range(6):
            mat = "PP" if e % 2 == 0 else "ABS"
            set_p = [100, 150, 200, 250, 300, 350][e]
            for c in range(10):
                cid += 1
                hp = set_p + rng.normal(0, 25)
                p = 1 / (1 + np.exp((hp - 190) / 20))
                nok = int(rng.random() < p)
                t = pd.Timestamp(day) + pd.Timedelta(minutes=e * 30 + c)
                rows.append({
                    "MET_JobCycleID": cid, "MET_MachineCycleID": 1000 + cid,
                    "MET_TimeCycleStart": t, "MET_TimeCycleEnd": t + pd.Timedelta(seconds=40),
                    "MET_ExperimentNumber": f"A{e + 1:02d}", "MET_MaterialName": mat,
                    "MET_MaterialMoisture": rng.normal(0.05, 0.01),
                    "SET_HoldingPressure": set_p,
                    "DXP_HoldingPressure1": hp,
                    "DXP_Noise": rng.normal(0, 1),
                    "CALC_InjectionStroke": rng.normal(140, 5) if mat == "PP" else np.nan,
                    "ENV_Temperature": rng.normal(22, 1),
                    "IR_Area": rng.normal(10, 1) + 3 * nok,          # 사후측정: 입력에서 빠져야 함
                    "SCA_Weight": rng.normal(30, 0.5) - nok,           # 사후측정
                    "LBL_SinkMarks": nok, "LBL_Underfilled": 0, "LBL_NOK": nok,
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def make_config(dst: Path) -> None:
    src = WF / "config"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.yaml"):
        shutil.copy2(f, dst / f.name)
    e = yaml.safe_load((dst / "experiment.yaml").read_text(encoding="utf-8"))
    e["data"]["expected_sha256"] = None
    e["data"].pop("fallback_csv", None)
    e["cv"] = {"n_splits": 3, "n_repeats": 2}
    e["wrapper"].update({"estimators": ["logistic_regression", "random_forest"], "k_grid": "1-4",
                         "param_overrides": {"random_forest": {"n_estimators": 30}}})
    e["model"]["candidates"] = ["logistic_regression", "random_forest"]
    e["model"]["calibration"] = {"method": "sigmoid", "cv": 2}
    e["model"]["params"]["random_forest"]["n_estimators"] = 30
    e["explain"]["permutation_repeats"] = 3
    (dst / "experiment.yaml").write_text(yaml.safe_dump(e, allow_unicode=True, sort_keys=False), encoding="utf-8")


class WorkflowEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)
        cls.data = TMP / "data" / "tiny.parquet"
        cls.config = TMP / "config"
        cls.runs = TMP / "runs"
        cls.release = TMP / "release"
        cls.share = TMP / "share"
        make_dataset(cls.data)
        make_config(cls.config)
        cls.env = {**os.environ, "ICT_CONFIG_DIR": str(cls.config), "ICT_DATA_PATH": str(cls.data),
                   "ICT_RUNS_DIR": str(cls.runs), "PYTHONIOENCODING": "utf-8"}
        cls.run_out = cls.py("run.py", "--run", "t1")

    @classmethod
    def py(cls, *args, check=True):
        r = subprocess.run([sys.executable, *args], cwd=WF, env=cls.env, capture_output=True, text=True,
                           encoding="utf-8")
        if check and r.returncode != 0:
            raise AssertionError(f"{args} 실패 ({r.returncode})\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        return r

    def test_1_all_stages_ok(self):
        m = json.loads((self.runs / "t1" / "manifest.json").read_text(encoding="utf-8"))
        for s in ["validate", "eda", "split", "filter", "wrapper", "model", "explain", "release", "report"]:
            self.assertEqual(m["stages"][s]["status"], "ok", s)

    def test_2_post_measurement_excluded(self):
        th = json.loads((self.runs / "t1" / "05_model" / "thresholds.json").read_text(encoding="utf-8"))
        for f in th["features"]:
            self.assertFalse(f.startswith(("IR_", "SCA_", "LBL_")), f)

    def test_3_bundle_contract(self):
        b = joblib.load(self.runs / "t1" / "07_release" / "model.joblib")
        for k in ["schema_version", "pipeline", "features", "alert_threshold", "caution_threshold",
                  "reference", "missing_policy", "units", "ranges", "version", "data_source"]:
            self.assertIn(k, b)
        self.assertLessEqual(b["caution_threshold"], b["alert_threshold"])
        X = pd.DataFrame([{f: b["reference"]["median"][f] for f in b["features"]}])
        p = b["pipeline"].predict_proba(X)[:, 1]
        self.assertTrue(0 <= p[0] <= 1)
        for f in b["missing_policy"]["optional"]:     # 선택 특징은 비어 있어도 예측돼야 함
            X2 = X.copy(); X2[f] = np.nan
            b["pipeline"].predict_proba(X2)

    def test_4_benchmark_keys_and_privacy(self):
        bench = json.loads((self.runs / "t1" / "07_release" / "benchmark_public.json").read_text(encoding="utf-8"))
        for k in ["version", "selected_model", "alert_threshold", "model_sha256", "validation_comparison",
                  "test", "test_all_risk", "test_all_normal", "feature_analysis", "quality_relation_mode"]:
            self.assertIn(k, bench)
        for k in ["recall", "precision", "f2", "average_precision", "alert_rate", "count", "tp"]:
            self.assertIn(k, bench["test"])
        for fa in bench["feature_analysis"]:
            for g in fa["quality_relation"]:
                self.assertGreaterEqual(g["count"], 5)
        text = json.dumps(bench)
        self.assertNotIn("1001", text.replace("1001.", ""))   # 부품 ID가 새지 않음 (MachineCycleID 1001~)

    def test_5_public_copy_has_no_private_files(self):
        pub = self.runs / "t1" / "public"
        self.assertTrue(pub.exists())
        for f in ["02_split/split.csv", "05_model/oof_predictions.csv", "06_explain/model.joblib",
                  "07_release/model.joblib", "07_release/real_demo.csv"]:
            self.assertFalse((pub / f).exists(), f)
        self.assertFalse(list(pub.rglob("_ckpt_*")))

    def test_6_promote_and_synthetic(self):
        chk = json.loads((self.runs / "t1" / "07_release" / "release_check.json").read_text(encoding="utf-8"))
        r = self.py("promote.py", "--run", "t1", "--target", str(self.release), "--local-target", str(TMP / "local"),
                    check=False)
        if chk["passed"]:
            self.assertEqual(r.returncode, 0, r.stdout)
        else:
            self.assertEqual(r.returncode, 1, r.stdout)
            self.py("promote.py", "--run", "t1", "--target", str(self.release), "--local-target", str(TMP / "local"),
                    "--force")
        info = json.loads((self.release / "release_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["files"]["model.joblib"], info["model_sha256"])
        local = TMP / "local"
        self.assertTrue((local / "real_demo.csv").exists() and (local / "real_demo_definition.json").exists())
        import hashlib
        self.assertEqual(info["real_demo"]["sha256"], hashlib.sha256((local / "real_demo.csv").read_bytes()).hexdigest())
        self.assertEqual(info["real_demo"]["rows"], len(pd.read_csv(local / "real_demo.csv")))
        self.py("promote.py", "--run", "t1", "--target", str(self.release), "--local-target", str(TMP / "local"),
                    "--force")
        self.assertTrue((self.release / "previous").exists())

        self.py("make_synthetic.py", "--release", str(self.release), "--out", str(self.share))
        demo = pd.read_csv(self.share / "synthetic_demo.csv")
        test = pd.read_csv(self.share / "test_upload.csv")
        self.assertEqual(len(demo), 120)
        self.assertTrue(demo["part_id"].is_unique)
        self.assertTrue(set(demo["actual_label"]) <= {0, 1})
        self.assertGreater(test["part_id"].duplicated().sum(), 0)    # 오류 행 포함

    def test_7_resume_refuses_finished(self):
        r = self.py("run.py", "--resume", "t1", "--stages", "model", check=False)
        self.assertNotEqual(r.returncode, 0)

    def test_8_to_service_input_and_missing_columns(self):
        sys.path.insert(0, str(WF))
        os.environ["ICT_CONFIG_DIR"] = str(self.config)
        try:
            from src.config import load_config
            from src.contract import check_service_input, service_columns, to_service_input
            from src.errors import StageError
            cfg = load_config()
        finally:
            os.environ.pop("ICT_CONFIG_DIR", None)
        dictionary = json.loads((self.runs / "t1" / "07_release" / "data_dictionary.json").read_text(encoding="utf-8"))
        raw = pd.read_parquet(self.data)
        out = to_service_input(raw, cfg, dictionary)
        self.assertEqual(list(out.columns), service_columns(dictionary))
        self.assertEqual(len(out), len(raw))
        self.assertTrue((out["data_source"] == dictionary["data_source"]).all())
        self.assertEqual(out["part_id"].iloc[0], str(raw["MET_MachineCycleID"].iloc[0]))
        self.assertRegex(out["cycle_start_at"].iloc[0], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        self.assertEqual(out["actual_label"].tolist(), raw["LBL_NOK"].tolist())
        for f in dictionary["features"]:   # 값은 그대로 (변환·대체 없음)
            pd.testing.assert_series_equal(out[f["name"]], raw[f["name"]].reset_index(drop=True), check_names=False)
        self.assertEqual(check_service_input(out, dictionary), [])
        with self.assertRaises(StageError) as e:
            to_service_input(raw.drop(columns=["MET_MachineCycleID"]), cfg, dictionary)
        self.assertIn("MET_MachineCycleID", str(e.exception))

    def test_9_make_upload_cli(self):
        rel = self.runs / "t1" / "07_release"
        out = TMP / "upload_holdout.csv"
        r = self.py("make_upload.py", "--input", str(self.data), "--release", str(rel), "--holdout-of", "t1",
                    "--output", str(out))
        self.assertNotIn("[주의]", r.stdout)
        split = pd.read_csv(self.runs / "t1" / "02_split" / "split.csv")
        df = pd.read_csv(out, dtype={"part_id": str})
        self.assertEqual(len(df), int((split["split"] == "holdout").sum()))
        r = self.py("make_upload.py", "--input", str(self.data), "--release", str(rel), "--output", str(TMP / "all.csv"))
        self.assertIn("[주의]", r.stdout)
        other = TMP / "other.parquet"
        pd.read_parquet(self.data).head(10).to_parquet(other)
        r = self.py("make_upload.py", "--input", str(other), "--release", str(rel), "--holdout-of", "t1",
                    "--output", str(TMP / "x.csv"), check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("SHA-256", r.stdout)

    def test_10_real_demo_generated_and_checked(self):
        rel = self.runs / "t1" / "07_release"
        split = pd.read_csv(self.runs / "t1" / "02_split" / "split.csv")
        demo = pd.read_csv(rel / "real_demo.csv", dtype={"part_id": str})
        dictionary = json.loads((rel / "data_dictionary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(demo), int((split["split"] == "holdout").sum()))
        self.assertEqual(list(demo.columns), [m["name"] for m in dictionary["meta"]]
                         + [f["name"] for f in dictionary["features"]] + [m["name"] for m in dictionary["optional"]])
        chk = json.loads((rel / "release_check.json").read_text(encoding="utf-8"))
        names = {c["check"]: c for c in chk["checks"]}
        self.assertTrue(names["실측 예제 형식 (서비스 검증 규칙)"]["passed"])
        self.assertTrue(names["실측 예제 판정 = holdout 결과 (TP/FN/FP/TN)"]["passed"])
        self.assertEqual(chk["real_demo"]["rows"], len(demo))
        self.assertTrue((self.runs / "t1" / "public" / "07_release" / "real_demo_definition.json").exists())

    def test_11_real_demo_mismatch_blocks_and_missing_column_warns(self):
        sys.path.insert(0, str(WF))
        os.environ["ICT_CONFIG_DIR"] = str(self.config); os.environ["ICT_DATA_PATH"] = str(self.data)
        try:
            from src.config import load_config
            from src.pipeline import load_split
            from src.runio import RunContext
            from stages import release
            cfg = load_config()
            ctx = RunContext(self.runs / "t1", cfg)
            _, hold = load_split(ctx)
        finally:
            os.environ.pop("ICT_CONFIG_DIR", None); os.environ.pop("ICT_DATA_PATH", None)
        rel = self.runs / "t1" / "07_release"
        dictionary = json.loads((rel / "data_dictionary.json").read_text(encoding="utf-8"))
        b = joblib.load(rel / "model.joblib")
        hold_m = pd.read_csv(self.runs / "t1" / "05_model" / "holdout_metrics.csv")
        hsel = hold_m[hold_m["selected"].astype(bool)].iloc[0].to_dict()
        hsel["TP"] += 1                                   # 서비스 판정과 어긋나게 만든다
        out = TMP / "rd_check"; out.mkdir(exist_ok=True)
        checks, warnings = [], []
        release._real_demo(ctx, cfg, cfg.release, dictionary, b["pipeline"], b["features"], b["alert_threshold"],
                           hold, hsel, {}, out, checks, warnings)
        self.assertFalse(next(c for c in checks if "판정" in c["check"])["passed"])
        broken = json.loads(json.dumps(dictionary)); broken["meta"][0]["source_column"] = "NO_SUCH_COLUMN"
        checks, warnings = [], []
        r = release._real_demo(ctx, cfg, cfg.release, broken, b["pipeline"], b["features"], b["alert_threshold"],
                               hold, hsel, {}, out, checks, warnings)
        self.assertIsNone(r); self.assertEqual(checks, [])
        self.assertTrue(any("NO_SUCH_COLUMN" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
