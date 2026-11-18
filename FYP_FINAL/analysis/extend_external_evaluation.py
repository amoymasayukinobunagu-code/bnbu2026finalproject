"""Evaluate the three additional full-data checkpoints on the audited external frames."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from segment_anything import SamPredictor, sam_model_registry

from run_external_and_speed import ROOT, WEIGHTS, file_hash, forward, prepared

ORIGINAL_RUN = ROOT / "analysis/outputs/external_box_20261118"
ORIGINAL_PROJECT = WEIGHTS.parent
MODELS = {
    "SAMed": (WEIGHTS / "sam_vit_b.pth",
              ORIGINAL_PROJECT / "results/samed_baseline_cloud/checkpoints/r4_full/samed_lora_r4_best.pth",
              "numbers/ablation_box_expand/samed_baseline/prompt_ablation_box_expand/box30/per_image.json"),
    "MedSAM+LoRA": (WEIGHTS / "medsam_vit_b.pth",
                   WEIGHTS / "medsam_peft/r4_full/medsam_lora_r4_best.pth",
                   "numbers/official_core/medsam_peft/r4_full_best/per_image.json"),
    "PP-SAM": (WEIGHTS / "sam_vit_b.pth",
               WEIGHTS / "pp_sam/pp_sam_vit_b_880shot_best.pth",
               "numbers/official_core/pp_sam_baseline_v3/eval_best_880shot/per_image.json"),
}
EXPECTED_COUNTS = {"CVC-ColonDB": 342, "ETIS-LaribPolypDB": 190}


def checkpoint_predictor(name, backbone, weights):
    sam = sam_model_registry["vit_b"](checkpoint=str(backbone))
    if name == "PP-SAM":
        state = torch.load(weights, map_location="cpu", weights_only=True)
        sam.load_state_dict(state, strict=True)
    else:
        sys.path.insert(0, str(ORIGINAL_PROJECT / "code/samed_baseline"))
        try:
            from lora_sam import LoRA_Sam
        finally:
            sys.path.pop(0)
        adapter = LoRA_Sam(sam, r=4)
        adapter.load_lora_parameters(str(weights))
        sam = adapter.sam
    return SamPredictor(sam.cuda().eval())


def audited_inputs():
    inputs = pd.read_csv(ORIGINAL_RUN / "duplicate_label_audit.csv", dtype={"id": str})
    included = inputs[inputs.include_in_reviewed_summary].copy()
    assert included.groupby("dataset").size().to_dict() == EXPECTED_COUNTS
    assert not included.duplicated(["dataset", "id"]).any()
    assert not included["sha256"].duplicated().any()
    original = pd.read_csv(ORIGINAL_RUN / "per_image.csv", dtype={"id": str})
    retained = original.merge(included[["dataset", "id"]], on=["dataset", "id"], validate="many_to_one")
    assert len(retained) == 2 * len(included)
    assert not retained.duplicated(["model", "dataset", "id"]).any()
    archived_boxes = retained[retained.model == "SAM-B"].set_index(["dataset", "id"]).box
    adapter_boxes = retained[retained.model == "MSA-B"].set_index(["dataset", "id"]).box
    assert archived_boxes.sort_index().equals(adapter_boxes.sort_index())
    inputs = []
    for row in included.itertuples(index=False):
        image_path, mask_path = Path(row.image), Path(row.mask)
        assert file_hash(image_path) == row.sha256, image_path
        rgb, gt, box = prepared(image_path, mask_path)
        assert hashlib.sha256(gt.tobytes()).hexdigest() == row.binary_mask_sha256, mask_path
        assert list(box) == ast.literal_eval(archived_boxes.loc[(row.dataset, row.id)]), image_path
        inputs.append((row.dataset, row.id, image_path, mask_path))
    return inputs, retained


def score(pred, rgb, gt, box):
    result = forward(pred, rgb, box)
    intersection = int((result & gt).sum())
    return 2 * intersection / (result.sum() + gt.sum()), intersection / (result | gt).sum()


def audit_checkpoint(pred, archive):
    existing = json.loads((ROOT / archive).read_text(encoding="utf-8"))
    reference = pd.read_csv(ROOT / "analysis/outputs/image_features.csv", dtype={"id": str})
    rows = []
    for dataset in ("Kvasir-SEG-val", "CVC-ClinicDB"):
        paths = reference[reference.dataset == dataset].set_index("id")
        saved = existing[dataset]["per_image"]
        for index in (0, len(saved) // 2, len(saved) - 1):
            expected = saved[index]
            stem = Path(expected["name"]).stem
            paths_for_frame = paths.loc[stem]
            rgb, gt, box = prepared(Path(paths_for_frame.image), Path(paths_for_frame["mask"]))
            dice, iou = score(pred, rgb, gt, box)
            assert abs(dice - expected["dice"]) <= 1e-6, (dataset, stem, dice, expected["dice"])
            assert abs(iou - expected["iou"]) <= 1e-6, (dataset, stem, iou, expected["iou"])
            rows.append(dict(dataset=dataset, id=stem, dice=dice, iou=iou,
                             archive_dice=expected["dice"], archive_iou=expected["iou"]))
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    inputs, original = audited_inputs()
    print(f"Verified image and mask hashes and box coordinates for {len(inputs)} external frames", flush=True)
    notes = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "prompt_expansion_pixels": 30,
        "screened_counts": EXPECTED_COUNTS,
        "source_run": ORIGINAL_RUN.relative_to(ROOT).as_posix(),
        "source_file_sha256": {name: file_hash(ORIGINAL_RUN / name) for name in
                               ("duplicate_label_audit.csv", "per_image.csv", "summary_reviewed.csv")},
        "checkpoints": {name: {key: {"path": str(path), "sha256": file_hash(path)}
                               for key, path in zip(("backbone", "adapted"), weights[:2])}
                        for name, weights in MODELS.items()},
    }
    (args.out / "run.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    results = [original]
    verification = []
    for model, (backbone, weights, archive) in MODELS.items():
        pred = checkpoint_predictor(model, backbone, weights)
        checked = audit_checkpoint(pred, archive)
        verification.extend(dict(model=model, archive=archive, **row) for row in checked)
        print(f"{model}: {len(checked)} archived-frame checks passed", flush=True)
        records = []
        for index, (dataset, stem, image_path, mask_path) in enumerate(inputs, 1):
            rgb, gt, box = prepared(image_path, mask_path)
            dice, iou = score(pred, rgb, gt, box)
            records.append(dict(model=model, dataset=dataset, id=stem, dice=dice, iou=iou, box=list(box)))
            if index % 100 == 0 or index == len(inputs):
                print(f"{model}: {index}/{len(inputs)} frames", flush=True)
        data = pd.DataFrame(records)
        data.to_csv(args.out / f"{model}_per_image.csv", index=False)
        results.append(data)
        del pred
        torch.cuda.empty_cache()
    combined = pd.concat(results, ignore_index=True)
    assert not combined.duplicated(["model", "dataset", "id"]).any()
    summary = combined.groupby(["model", "dataset"]).agg(n=("dice", "size"), dice=("dice", "mean"),
                                                         iou=("iou", "mean")).reset_index()
    assert all(dict(zip(group.dataset, group.n)) == EXPECTED_COUNTS for _, group in summary.groupby("model"))
    combined.to_csv(args.out / "per_image.csv", index=False)
    summary.to_csv(args.out / "summary_reviewed.csv", index=False)
    pd.DataFrame(verification).to_csv(args.out / "checkpoint_verification.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
