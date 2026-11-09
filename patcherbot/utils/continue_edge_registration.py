"""Offline dark-edge registration. Run this file without arguments.

Combines original-image edge matches with relative recorded stage movement.
The optical scale stays fixed; stage priors constrain ambiguous placements.
No fitted reference mosaic or application calibration is modified.
"""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from skimage.registration import phase_cross_correlation
import tifffile


class EdgeRegistration:
    DATA = Path(__file__).resolve().parents[2] / "experiments" / "Data"
    OUTPUT = DATA / "scan_stitch_data" / "2026_11_05-11_47"
    WINDOW40 = (1793897624.464, 1793897751.718)
    WINDOW4 = (1793827342.7041066, 1793827605.1501741)
    STRIDE40 = 6
    WORKERS = 8
    CACHE_VERSION = 1
    TARGET_ERROR_UM = 2.0  # Maximum independent return-visit XY error requested by user.

    def __init__(self):
        cv2.setNumThreads(1)
        self.fit = json.loads((self.OUTPUT / "pure_dark_edge_fit.json").read_text())
        self.paths40 = self.catalog("2026_11_05-11_47", self.WINDOW40)
        self.paths4 = self.catalog("2026_11_04-16_20", self.WINDOW4)
        prior = json.loads((self.OUTPUT / "all_frame_registration_matches.json").read_text())
        self.ids4 = sorted({126, 132, 137, 147} | {
            p["frame4_index"] for f in prior["frames"] for p in f["pairs"]
            if p["usable_directional"]})
        self.ids40 = sorted(set(range(0, len(self.paths40), self.STRIDE40)) |
                            set(self.fit["frame_ids"]))

    def catalog(self, session, window):
        folder = self.DATA / "rig_recorder_data" / session / "camera_frames"
        return [p for t, p in sorted(
            (float(p.stem.rsplit("_", 1)[1]), p) for p in folder.glob("*.webp"))
                if window[0] <= float(p.stem.rsplit("_", 1)[1]) <= window[1]]

    @staticmethod
    def edges(image, side, largest):
        g = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), 1.2)
        lo, hi = np.percentile(g, [3, 97])
        g = np.clip((g - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
        dark = cv2.dilate((g < 120).astype(np.uint8), np.ones((5, 5), np.uint8))
        edge = ((cv2.Canny(g, 25, 65) > 0) & (dark > 0)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(edge, 8)
        if n < 2:
            return np.zeros_like(edge)
        keep = ([1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])] if largest else
                np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 20))
        keep = [k for k in keep if k != 0]
        return np.isin(labels, keep).astype(np.uint8)

    @staticmethod
    def register(a, b, minimum):
        result = {"accepted": False}
        if min(a.sum(), b.sum()) < 40:
            return result
        mask = np.ones(a.shape, bool)
        shift, _, _ = phase_cross_correlation(
            cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1),
            cv2.GaussianBlur(b.astype(np.float32), (0, 0), 1),
            reference_mask=mask, moving_mask=mask, overlap_ratio=.35)
        d = shift[::-1]
        dx, dy = np.rint(d).astype(int)
        h, w = a.shape
        aa = a[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
        bb = b[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
        if aa.size < 100 or min(aa.sum(), bb.sum()) < 40:
            return result
        da = cv2.distanceTransform(1 - aa, cv2.DIST_L2, 5)
        db = cv2.distanceTransform(1 - bb, cv2.DIST_L2, 5)
        distances = np.r_[da[bb > 0], db[aa > 0]]
        fraction = float(np.mean(distances <= 2))
        result.update(delta=d.tolist(), fraction_within2=fraction,
                      median=float(np.median(distances)), accepted=fraction >= minimum)
        return result

    def links(self):
        cache = self.OUTPUT / "edge_continuation_links.json"
        signature = {"version": self.CACHE_VERSION, "ids4": self.ids4, "ids40": self.ids40}
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved["signature"] == signature:
                print("Reusing saved image-only links", flush=True)
                return saved["links4"], saved["links40"]
        def extract(item):
            objective, i = item
            paths = self.paths4 if objective == 4 else self.paths40
            return objective, i, self.edges(cv2.imread(str(paths[i]), 0),
                                            640 if objective == 4 else 320, objective == 40)
        with ThreadPoolExecutor(self.WORKERS) as pool:
            extracted = list(pool.map(extract, [(4, i) for i in self.ids4] +
                                               [(40, i) for i in self.ids40]))
        edges = {(o, i): e for o, i, e in extracted}
        jobs = [(4, i, j) for i, j in combinations(self.ids4, 2)] + [
            (40, self.ids40[k], self.ids40[k + d]) for k in range(len(self.ids40))
            for d in (1, 2) if k + d < len(self.ids40)]
        def work(job):
            o, i, j = job
            e = self.register(edges[o, i], edges[o, j], .75 if o == 4 else .95)
            e.update(objective=o, i=i, j=j)
            if e["accepted"] and o == 4:
                coarse = self.register(cv2.resize(edges[o, i], (320, 320),
                    interpolation=cv2.INTER_NEAREST), cv2.resize(edges[o, j], (320, 320),
                    interpolation=cv2.INTER_NEAREST), .65)
                e["multiscale_error"] = (float(np.linalg.norm(np.array(e["delta"]) -
                    2 * np.array(coarse["delta"]))) if coarse["accepted"] else None)
                e["accepted"] = coarse["accepted"] and e["multiscale_error"] <= 3
            return e
        with ThreadPoolExecutor(self.WORKERS) as pool:
            records = []
            for n, e in enumerate(pool.map(work, jobs), 1):
                records.append(e)
                if n % 100 == 0:
                    print(f"Image links {n}/{len(jobs)}", flush=True)
        links4 = [e for e in records if e["objective"] == 4]
        links40 = [e for e in records if e["objective"] == 40]
        cache.write_text(json.dumps(dict(signature=signature, links4=links4,
                                         links40=links40), indent=2, allow_nan=False))
        return links4, links40

    @staticmethod
    def solve(links, anchor, tolerance):
        edges = [dict(e) for e in links if e["accepted"]]
        component = {anchor}
        def connected():
            c = {anchor}
            while True:
                before = len(c)
                for e in edges:
                    if e["i"] in c or e["j"] in c:
                        c.update([e["i"], e["j"]])
                if len(c) == before:
                    return c
        # Every nonterminal pass removes at least one link. Always solve again
        # after pruning, including when pruning disconnects part of the graph.
        for _ in range(len(edges) + 1):
            component = connected()
            ids = sorted(component)
            index = {i: k for k, i in enumerate(ids)}
            selected = [e for e in edges if e["i"] in component and e["j"] in component]
            rows, cols, values, rhs = [], [], [], []
            for k, e in enumerate(selected):
                rows.extend([k, k]); cols.extend([index[e["i"]], index[e["j"]]])
                values.extend([-1., 1.]); rhs.append(e["delta"])
            rows.append(len(selected)); cols.append(index[anchor]); values.append(100.)
            rhs.append([0., 0.])
            design = coo_matrix((values, (rows, cols)),
                                shape=(len(selected) + 1, len(ids))).tocsr()
            pos = np.column_stack([lsqr(design, np.array(rhs)[:, a], atol=1e-8,
                                       btol=1e-8)[0] for a in range(2)])
            rejected = []
            for e in selected:
                e["cycle_residual"] = float(np.linalg.norm(pos[index[e["j"]]] -
                    pos[index[e["i"]]] - e["delta"]))
                if e["cycle_residual"] > tolerance:
                    rejected.append((e["i"], e["j"]))
            if not rejected:
                break
            edges = [e for e in edges if (e["i"], e["j"]) not in rejected]
        return {i: pos[index[i]] for i in ids}, selected

    def run(self):
        l4, l40 = self.links()
        p4, e4 = self.solve(l4, 126, 4.)
        p40, e40 = self.solve(l40, 185, 3.)
        p4 = {i: p * 2 for i, p in p4.items()}
        transform = np.array(self.fit["best"]["quarter_chain_to_original4x"])
        origin = np.array(self.fit["chain_origin_quarter_native40"])
        ref_images = {i: cv2.imread(str(self.paths4[i]), 0) for i in p4}
        ref_edges = {i: self.edges(g, 1280, False) for i, g in ref_images.items()}
        distances = {i: cv2.distanceTransform(1 - e, cv2.DIST_L2, 5)
                     for i, e in ref_edges.items()}
        accepted, checks = {}, []
        for i, p in p40.items():
            raw = cv2.imread(str(self.paths40[i]), 0)
            edge = self.edges(raw, 320, True)
            yy, xx = np.where(edge > 0)
            points = (np.column_stack([xx, yy]) + p - origin) @ transform[:, :2].T + transform[:, 2]
            candidates = []
            for j, offset in p4.items():
                local = points - offset
                inside = np.all((local >= 0) & (local < 1280), axis=1)
                if len(local) < 40 or inside.mean() < .95:
                    continue
                q = local[inside].astype(np.float32)
                d = cv2.remap(distances[j], q[:, :1], q[:, 1:], cv2.INTER_LINEAR).ravel()
                candidates.append(dict(frame4=j, fraction=float(np.mean(d <= 2)),
                                       median=float(np.median(d)), p90=float(np.percentile(d, 90))))
            best = max(candidates, key=lambda c: c["fraction"]) if candidates else None
            good = best is not None and best["fraction"] > .85
            checks.append(dict(frame40=i, accepted=good, best_original4x_check=best))
            if good:
                accepted[i] = np.column_stack([transform[:, :2] / 4,
                    transform[:, :2] @ (p - origin - .375) + transform[:, 2]])
        self.render(p4, ref_images, accepted)
        report = dict(status="image_only_edge_extension_candidate", script=Path(__file__).name,
            connected_original4x_frames=len(p4), connected_original40x_frames=len(p40),
            accepted_original40x_frames=sorted(accepted), checks=checks,
            source4x_poses={str(i): p.tolist() for i, p in p4.items()},
            source40x_transforms={str(i): m.tolist() for i, m in accepted.items()},
            source4x_paths={str(i): str(self.paths4[i]) for i in p4},
            source40x_paths={str(i): str(self.paths40[i]) for i in accepted},
            image_links4=e4, image_links40=e40,
            scale_native4_to_native40=1/self.fit["best"]["scale"],
            no_stage_or_stitched_reference_used=True,
            limitation="Only the image-connected component and frames passing original4x-edge checks are rendered. Smooth contours remain ambiguous along their tangent.")
        (self.OUTPUT / "edge_continuation_result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False))
        print(f"RESULT: {len(p4)} original4x, {len(p40)} connected40x, "
              f"{len(accepted)} accepted40x", flush=True)
        self.register_all_sources()

    def register_all_sources(self):
        """Find a raw4x counterpart and pixel mapping for every raw40x image.

        Search every shortlisted raw4x image coarsely, then refine the best
        three at original4x resolution. Reference selection and placement use
        dark contours only. No fitted mosaic is involved in this search.
        """
        started = time.perf_counter()
        scale = float(self.fit["best"]["scale"])
        if abs(self.fit["best"]["angle"]) > 1e-9:
            raise ValueError("This direct search currently requires the measured zero rotation")
        coarse_side, fine_side = round(640 * scale), round(1280 * scale)
        references = {}
        for j in self.ids4:
            raw = cv2.imread(str(self.paths4[j]), 0)
            fields = []
            for side in (640, 1280):
                e = self.edges(raw, side, False)
                fields.append(cv2.distanceTransform(1 - e, cv2.DIST_L2, 5))
            references[j] = fields

        def locate(distance, template):
            scores = cv2.matchTemplate(np.minimum(distance, 20), template,
                                      cv2.TM_CCORR) / max(float(template.sum()), 1e-6)
            _, _, location, _ = cv2.minMaxLoc(scores)
            x, y = location
            value = float(scores[y, x])
            excluded = scores.copy()
            excluded[max(0, y-20):y+21, max(0, x-20):x+21] = np.inf
            return location, value, float(excluded.min() - value)

        def process(i):
            raw = cv2.imread(str(self.paths40[i]), 0)
            e = self.edges(raw, 320, True)
            result = dict(frame40_index=i, original40x_path=str(self.paths40[i]),
                          contour_pixel_count=int(e.sum()), candidates=[])
            if e.sum() < 50:
                result["status"] = "insufficient_dark_edge"
                return result
            coarse = cv2.resize(e.astype(np.float32), (coarse_side, coarse_side),
                                interpolation=cv2.INTER_AREA)
            ranked = []
            for j, (d, _) in references.items():
                _, value, _ = locate(d, coarse)
                ranked.append((value, j))
            fine = cv2.resize(e.astype(np.float32), (fine_side, fine_side),
                              interpolation=cv2.INTER_AREA)
            yy, xx = np.where(e > 0)
            # Exact center convention for native1280 ->320 -> fine_side.
            source_native = 4 * (np.column_stack([xx, yy]) + .5) - .5
            effective_scale = fine_side / 1280
            for _, j in sorted(ranked)[:3]:
                distance = references[j][1]
                (x, y), value, margin = locate(distance, fine)
                shift = np.array([x, y]) + (effective_scale - 1) / 2
                matrix = np.array([[effective_scale, 0, shift[0]],
                                   [0, effective_scale, shift[1]], [0, 0, 1.]])
                points = source_native * effective_scale + shift
                residuals = cv2.remap(distance, points[:, :1].astype(np.float32),
                                     points[:, 1:].astype(np.float32), cv2.INTER_LINEAR).ravel()
                fraction = float(np.mean(residuals <= 2))
                median = float(np.median(residuals))
                p90 = float(np.percentile(residuals, 90))
                good = fraction >= .95 and median <= .75 and p90 <= 2
                unique = good and margin >= .5
                result["candidates"].append(dict(
                    original4x_index=j, original4x_path=str(self.paths4[j]),
                    mean_edge_distance=value, median_edge_distance=median,
                    p90_edge_distance=p90, fraction_within2_pixels=fraction,
                    spatial_uniqueness_margin=margin,
                    status="unique_edge_candidate" if unique else
                           "ambiguous_edge_candidate" if good else "poor_edge_match",
                    original40x_to_original4x=matrix.tolist(),
                    original4x_to_original40x=np.linalg.inv(matrix).tolist(),
                    source40x_footprint_in_original4x=(matrix @ np.array(
                        [[0, 1279, 0, 1279], [0, 0, 1279, 1279], [1, 1, 1, 1]])
                        )[:2].T.tolist()))
            result["candidates"].sort(key=lambda c: c["mean_edge_distance"])
            best = result["candidates"][0]
            result["status"] = best["status"]
            result["best_reference_index"] = best["original4x_index"]
            return result

        results = []
        with ThreadPoolExecutor(self.WORKERS) as pool:
            for n, result in enumerate(pool.map(process, range(len(self.paths40))), 1):
                results.append(result)
                if n % 200 == 0:
                    print(f"Raw pixel transforms {n}/{len(self.paths40)} "
                          f"({time.perf_counter()-started:.1f}s)", flush=True)
        counts = {s: sum(r["status"] == s for r in results)
                  for s in sorted({r["status"] for r in results})}
        report = dict(
            status="all_original40x_frames_searched",
            processed_original40x_count=len(results),
            searched_original4x_count=len(self.ids4), status_counts=counts,
            reference_shortlist="Original filenames from earlier raw-frame searches; no positions reused",
            coordinate_convention="Column vector [x,y,1]; original camera pixel centers, top-left=(0,0)",
            shared_scale_native40x_to_native4x=fine_side / 1280,
            shared_scale_native4x_to_native40x=1280 / fine_side,
            rotation_degrees=0., elapsed_seconds=time.perf_counter()-started,
            no_stage_or_stitched_reference_used=True,
            validation_limit="Per-frame dark-contour candidates, not certified scene identity. "
                "A straight edge cannot determine position along its tangent. All candidate matrices "
                "are retained, but ambiguous/poor matches must not be used as calibrated mappings. "
                "Inverse mappings apply only inside the mapped40x footprint in the named4x image.",
            frames=results)
        destination = self.OUTPUT / "all_source_pixel_transforms.json"
        destination.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"ALL SOURCE RESULTS: {counts}", flush=True)
        self.source_pair_preview(results)

    def source_pair_preview(self, results):
        """Show an evenly spaced sample of the candidates, including failures."""
        tiles = []
        for i in np.linspace(0, len(results)-1, 12).astype(int):
            item = results[i]
            if not item["candidates"]:
                continue
            best = item["candidates"][0]
            raw40 = cv2.imread(item["original40x_path"], 0)
            raw4 = cv2.imread(best["original4x_path"], 0)
            matrix = np.array(best["original40x_to_original4x"])
            # Sample the exact original4x area corresponding to the40x field.
            u = (np.arange(240) + .5) * 1280 / 240 - .5
            xx, yy = np.meshgrid(u, u)
            mx = (matrix[0, 0] * xx + matrix[0, 2]).astype(np.float32)
            my = (matrix[1, 1] * yy + matrix[1, 2]).astype(np.float32)
            crop = cv2.remap(raw4, mx, my, cv2.INTER_LINEAR)
            pair = np.hstack([cv2.resize(raw40, (240, 240)), crop])
            tile = cv2.cvtColor(pair, cv2.COLOR_GRAY2RGB)
            tile = cv2.copyMakeBorder(tile, 44, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(tile, f"40x {i} -> raw4x {best['original4x_index']}",
                        (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
            cv2.putText(tile, item["status"], (5, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
            tiles.append(tile)
        while tiles and len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        if tiles:
            Image.fromarray(np.vstack([np.hstack(tiles[k:k+3])
                for k in range(0, len(tiles), 3)])).save(
                    self.OUTPUT / "all_source_pixel_transform_examples.png")

    def render(self, p4, refs, transforms):
        low = np.floor(np.array(list(p4.values())).min(0)).astype(int)
        high = np.ceil(np.array(list(p4.values())).max(0) + 1280).astype(int)
        width, height = high - low
        if width * height > 100_000_000:
            raise RuntimeError("Image graph produced an implausibly large canvas")
        acc4 = np.zeros((height, width), np.float32); den4 = np.zeros_like(acc4)
        acc40 = np.zeros_like(acc4); den40 = np.zeros_like(acc4)
        def weights(n):
            x = np.minimum(np.arange(n) + 1, np.arange(n, 0, -1)).astype(np.float32)
            return (x[:, None] * x[None, :]) / (x.max() ** 2)
        w4 = weights(1280)
        for i, g in refs.items():
            x, y = np.rint(p4[i] - low).astype(int)
            acc4[y:y+1280, x:x+1280] += g * w4; den4[y:y+1280, x:x+1280] += w4
        ds = round(1280 * self.fit["best"]["scale"]); wt = weights(ds)
        for i, matrix in transforms.items():
            g = cv2.resize(cv2.imread(str(self.paths40[i]), 0), (ds, ds), interpolation=cv2.INTER_AREA)
            m = matrix.copy()
            m[:, 2] += m[:, :2] @ np.full(2, .5 / (ds / 1280) - .5)
            m[:, :2] /= ds / 1280; m[:, 2] -= low
            acc40 += cv2.warpAffine(g * wt, m, (width, height))
            den40 += cv2.warpAffine(wt, m, (width, height))
        bg = np.clip(acc4 / np.maximum(den4, 1e-6), 0, 255).astype(np.uint8)
        hi = np.clip(acc40 / np.maximum(den40, 1e-6), 0, 255).astype(np.uint8)
        image = np.repeat(bg[:, :, None], 3, 2); mask = den40 > .01
        image[mask] = np.stack([bg[mask], hi[mask], hi[mask]], 1)
        tifffile.imwrite(self.OUTPUT / "edge_continuation_overlay.tif", image,
                         photometric="rgb", compression="deflate")
        preview = Image.fromarray(image); preview.thumbnail((1600, 1400))
        preview.save(self.OUTPUT / "edge_continuation_overlay.png")

    def joint_stage_image_registration(self):
        """Robust translation graph with micron priors; retain every source frame."""
        matches = json.loads((self.OUTPUT / "all_source_pixel_transforms.json").read_text())
        cached = json.loads((self.OUTPUT / "edge_continuation_links.json").read_text())
        def stage(session, paths):
            data = np.genfromtxt(self.DATA / "rig_recorder_data" / session /
                                "movement_recording.csv", delimiter=";", names=True)
            times = np.array([float(p.stem.rsplit("_", 1)[1]) for p in paths])
            return np.column_stack([np.interp(times, data["timestamp"], data[k])
                                    for k in ("st_x", "st_y")])
        s4 = stage("2026_11_04-16_20", self.paths4)
        s40 = stage("2026_11_05-11_47", self.paths40)
        calibration = self.OUTPUT.parents[3] / "patcherbot/utils/stage_calibration_4x.json"
        b4 = -np.array(json.loads(calibration.read_text())["stage"]["M"])
        ids4 = self.ids4; n4 = len(ids4); n40 = len(self.paths40)
        ix4 = {j: k for k, j in enumerate(ids4)}
        prior4 = (s4[ids4] - s4[126]) @ b4.T
        relative40 = (s40 - s40[185]) @ b4.T
        offsets = []
        for f in matches["frames"]:
            if (f["frame40_index"] // 60) % 5 == 0:
                continue
            votes = [prior4[ix4[c["original4x_index"]]] +
                     np.array(c["original40x_to_original4x"])[:2, 2] -
                     relative40[f["frame40_index"]] for c in f["candidates"]
                     if c["status"] == "unique_edge_candidate"]
            if votes:
                offsets.append(np.median(votes, axis=0))
        offset = np.median(offsets, axis=0)
        prior40 = relative40 + offset
        initial = np.vstack([prior4, prior40])
        constraints = []
        validation = []
        def add(terms, target, sigma, kind, heldout=False):
            item = (terms, np.asarray(target), sigma, kind)
            (validation if heldout else constraints).append(item)
        for k in range(n4+n40):
            add([(k, 1)], initial[k], 2 if k < n4 else 4, "stage_absolute")
        add([(ix4[126], 1)], [0, 0], .01, "origin")
        for i in range(n40-1):
            add([(n4+i+1, 1), (n4+i, -1)], relative40[i+1]-relative40[i],
                1.5, "stage_increment")
        for objective, factor, stage_positions in [(4, 2, prior4), (40, .4, prior40)]:
            for e in cached["links"+str(objective)]:
                if not e["accepted"]:
                    continue
                i, j = e["i"], e["j"]
                a, b = (ix4[i], ix4[j]) if objective == 4 else (n4+i, n4+j)
                target = np.array(e["delta"]) * factor
                # Reject image links incompatible with the recorded trajectory.
                if np.linalg.norm(initial[b]-initial[a]-target) > (35 if objective == 4 else 12):
                    continue
                heldout = (i // 60) % 5 == 0 if objective == 40 else (i+j) % 5 == 0
                add([(b, 1), (a, -1)], target, 1.0, "image_neighbor_"+str(objective), heldout)
        image_supported = set()
        for f in matches["frames"]:
            i = f["frame40_index"]
            candidates = []
            for c in f["candidates"]:
                if c["fraction_within2_pixels"] < .95 or c["mean_edge_distance"] > 1.0:
                    continue
                j = ix4[c["original4x_index"]]
                target = np.array(c["original40x_to_original4x"])[:2, 2]
                error = np.linalg.norm(prior40[i]-prior4[j]-target)
                if error < (30 if c["status"] == "unique_edge_candidate" else 12):
                    candidates.append((error, j, target, c))
            if not candidates:
                continue
            _, j, target, candidate = min(candidates, key=lambda c: c[0])
            heldout = (i // 60) % 5 == 0
            add([(n4+i, 1), (j, -1)], target,
                1.0 if candidate["status"] == "unique_edge_candidate" else 3.0,
                "image_cross_objective", heldout)
            if not heldout:
                image_supported.add(i)
        rr=[]; cc=[]; vv=[]
        for row, (terms, target, sigma, kind) in enumerate(constraints):
            for col, value in terms:
                rr.append(row); cc.append(col); vv.append(value)
        a = coo_matrix((vv, (rr, cc)), shape=(len(constraints), n4+n40)).tocsr()
        target = np.array([c[1] for c in constraints]); sigma = np.array([c[2] for c in constraints])
        solved = initial.copy()
        for iteration in range(8):
            residual = np.linalg.norm(a @ solved-target, axis=1)
            robust = np.minimum(1, 2.5*sigma/np.maximum(residual, 1e-9))
            weights = np.sqrt(robust)/sigma
            weighted = a.multiply(weights[:, None]).tocsr()
            solved = np.column_stack([lsqr(weighted, target[:, k]*weights,
                atol=1e-8, btol=1e-8, iter_lim=3000)[0] for k in (0, 1)])
        def errors(items, positions):
            return np.array([np.linalg.norm(sum(positions[k]*v for k,v in terms)-target)
                             for terms,target,_,_ in items])
        checks = {}
        for kind in sorted({c[3] for c in validation}):
            items = [c for c in validation if c[3] == kind]
            before = errors(items, initial); after = errors(items, solved)
            checks[kind] = dict(count=len(items), median_before=float(np.median(before)),
                median_after=float(np.median(after)), p90_after=float(np.percentile(after,90)))
        for f in matches["frames"]:
            i = f["frame40_index"]
            # Pick a nearby original reference; the solved global position is unchanged.
            j = int(np.argmin(np.linalg.norm(solved[:n4]+640-(solved[n4+i]+64), axis=1)))
            matrix = np.eye(3); matrix[:2,:2] *= .1
            matrix[:2,2] = solved[n4+i]-solved[j]
            f["candidates"] = [dict(original4x_index=ids4[j],
                original40x_to_original4x=matrix.tolist())]
            f["status"] = "joint_image_stage" if i in image_supported else "stage_and_neighbor_constraints"
        report = dict(matches=matches, positions={str(j):solved[ix4[j]].tolist() for j in ids4},
            optical_scale=.1, stage_microns_to_4x_pixels=b4.tolist(), anchor_translation=offset.tolist(),
            training_constraints=len(constraints), directly_image_supported_frames=len(image_supported),
            heldout_validation_4x_pixels=checks,
            validation_limit="Held-out constraints come from existing candidate matches, not independent ground truth. "
                "Temporal blocks exclude cross-objective/40x image constraints. Optical scale is fixed.",
            max_adjacent_placement_step=float(np.linalg.norm(np.diff(solved[n4:],axis=0),axis=1).max()),
            max_adjacent_stage_correction_step=float(np.linalg.norm(np.diff(solved[n4:]-prior40,axis=0),axis=1).max()))
        (self.OUTPUT / "joint_stage_image_solution.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ("matches","positions")}, indent=2), flush=True)
        self.render_every_source(report)

    def save_joint_anchor(self):
        """Persist the shared curved boundary as an image-template coordinate origin."""
        import hashlib
        report_path=self.OUTPUT/"restored_joint_source_registration.json"
        report=json.loads(report_path.read_text());f40=report["frame40"][214]
        f4=report["frame4"][f40["reference4_index"]]
        image40=cv2.imread(f40["path"],0);image4=cv2.imread(f4["path"],0)
        blur=cv2.GaussianBlur(image40.astype(np.float32),(0,0),3)
        derivative=cv2.Sobel(blur,cv2.CV_32F,0,1)
        xs=np.arange(100,1180);ys=180+np.argmax(derivative[180:700,100:1180],axis=0)
        smooth=np.convolve(ys,np.ones(61)/61,mode="valid");middle=xs[30:-30][np.argmin(smooth)]
        keep=np.abs(xs-middle)<=200;coef=np.polyfit(xs[keep]-middle,ys[keep],2)
        vertex=-coef[1]/(2*coef[0]);anchor40=np.array([middle+vertex,np.polyval(coef,vertex)])
        if coef[0]<=0 or abs(vertex)>200:raise RuntimeError("Curved-boundary anchor is not a stable interior minimum")
        local=np.array(f40["pixel40_to_reference4"]);global40=np.array(f40["pixel40_to_global4"])
        anchor4=(local@np.r_[anchor40,1])[:2];global_anchor=(global40@np.r_[anchor40,1])[:2]
        physical=np.eye(3);physical[:2,:2]=np.linalg.inv(-np.array(report["calibration"]["M4"]))
        physical[:2,2]=-physical[:2,:2]@global_anchor
        references={};panels=[]
        for objective,frame,g,point,wh in [(4,f4,image4,anchor4,(240,160)),(40,f40,image40,anchor40,(400,240))]:
            wh=np.array(wh);start=np.maximum(0,np.minimum(np.floor(point-wh/2).astype(int),np.array(g.shape[::-1])-wh))
            crop=g[start[1]:start[1]+wh[1],start[0]:start[0]+wh[0]]
            template=self.OUTPUT/f"restored_joint_anchor_{objective}x.png"
            Image.fromarray(crop).save(template)
            references[str(objective)]=dict(frame_index=frame["index"],source_path=frame["path"],
                source_sha256=hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest(),
                template_path=str(template),crop_xywh=[*start.tolist(),*wh.tolist()],
                anchor_pixel=point.tolist(),anchor_in_template=(point-start).tolist(),
                localization="detected lower-boundary crest" if objective==40 else "mapped from40x using saved joint transform")
            display_scale=min(600/wh[0],400/wh[1]);display_size=np.rint(wh*display_scale).astype(int)
            offset=(np.array([600,400])-display_size)//2;panel=np.zeros((400,600,3),np.uint8)
            dx,dy=offset;dw,dh=display_size
            panel[dy:dy+dh,dx:dx+dw]=cv2.cvtColor(cv2.resize(crop,(dw,dh)),cv2.COLOR_GRAY2BGR)
            xy=np.rint((point-start)*display_scale+offset).astype(int)
            cv2.drawMarker(panel,tuple(xy),(0,190,255),cv2.MARKER_CROSS,24,2)
            cv2.putText(panel,f"{objective}x original pixels; "+("mapped" if objective==4 else "detected"),
                (12,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,190,255),2);panels.append(panel)
        anchor=dict(status="reusable_image_template_anchor",feature="Crest of lower dark-to-bright boundary on shared wavy rim",
            source_map=str(report_path),source_map_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            references=references,global4_anchor_pixel=global_anchor.tolist(),
            global4_to_anchor_microns=physical.tolist(),anchor_microns_to_global4=np.linalg.inv(physical).tolist(),
            pixel40_to_pixel4=local.tolist(),pixel4_to_pixel40=np.linalg.inv(local).tolist(),
            boundary_fit_rms40_pixels=float(np.sqrt(np.mean((np.polyval(coef,xs[keep]-middle)-ys[keep])**2))),
            scale_native40_to_native4=.1,objective_calibrations=report["calibration"],
            limitation="Reloadable image anchor; physical permanence and2um return accuracy unverified. Microns use provisional common-map calibration.")
        assert np.linalg.norm((physical@np.r_[global_anchor,1])[:2])<1e-9
        (self.OUTPUT/"restored_joint_anchor.json").write_text(json.dumps(anchor,indent=2,allow_nan=False))
        cv2.imwrite(str(self.OUTPUT/"restored_joint_anchor_preview.png"),np.hstack(panels))
        return anchor

    def anchor_coordinates(self, pixels, frame_index, objective=40):
        """Original-frame pixel(s) to saved anchor-relative common-map microns."""
        if objective not in (4,40):raise ValueError("objective must be4 or40")
        anchor=json.loads((self.OUTPUT/"restored_joint_anchor.json").read_text())
        report=json.loads(Path(anchor["source_map"]).read_text())
        frame=report[f"frame{objective}"][frame_index]
        key="pixel4_to_global4" if objective==4 else "pixel40_to_global4"
        matrix=np.array(anchor["global4_to_anchor_microns"])@np.array(frame[key])
        points=np.asarray(pixels,dtype=float)
        return points@matrix[:2,:2].T+matrix[:2,2]

    def locate_joint_anchor(self, image, objective):
        """Re-find the stored template at the same objective; return image coordinates only."""
        if objective not in (4,40):raise ValueError("objective must be4 or40")
        anchor=json.loads((self.OUTPUT/"restored_joint_anchor.json").read_text())
        reference=anchor["references"][str(objective)]
        template=cv2.imread(reference["template_path"],0)
        source=image if isinstance(image,np.ndarray) else cv2.imread(str(image),0)
        if source is None:raise ValueError(f"Cannot read {image}")
        if source.ndim!=2 or source.dtype!=np.uint8:raise ValueError("Expected native uint8 grayscale pixels")
        if np.any(np.array(source.shape)<template.shape):raise ValueError("Image is smaller than anchor template")
        def gradient(g):
            g=cv2.GaussianBlur(g.astype(np.float32),(0,0),2)
            return cv2.Sobel(g,cv2.CV_32F,0,1)
        scores=cv2.matchTemplate(gradient(source),gradient(template),cv2.TM_CCOEFF_NORMED)
        _,score,_,xy=cv2.minMaxLoc(scores);position=np.array(xy,dtype=float)
        for axis,index in enumerate(xy):
            if 0<index<scores.shape[1-axis]-1:
                x,y=xy;v=scores[y,x-1:x+2] if axis==0 else scores[y-1:y+2,x]
                d=float(v[0]-2*v[1]+v[2])
                if d< -1e-7:position[axis]+=np.clip(.5*(v[0]-v[2])/d,-.5,.5)
        second=scores.copy();x,y=xy;second[max(0,y-8):y+9,max(0,x-8):x+9]=-1
        boundary=x==0 or y==0 or x==scores.shape[1]-1 or y==scores.shape[0]-1
        matched=score>=.85 and score-second.max()>.005 and not boundary
        return dict(anchor_pixel=(position+reference["anchor_in_template"]).tolist(),
            template_correlation=float(score),separated_peak_margin=float(score-second.max()),
            matched=bool(matched),search_boundary=bool(boundary),objective=objective,
            image_size=list(source.shape[::-1]),applied_to_hardware=False)

    def register_anchor_pair(self, image4, image40):
        """Map native4x/40x pixels through the shared anchor; keep saved scale/orientation."""
        import hashlib
        anchor_path=self.OUTPUT/"restored_joint_anchor.json"
        anchor_bytes=anchor_path.read_bytes();anchor=json.loads(anchor_bytes)
        found4=self.locate_joint_anchor(image4,4);found40=self.locate_joint_anchor(image40,40)
        if not found4["matched"] or not found40["matched"]:
            raise ValueError("Shared anchor was not confidently found in both images")
        point4=np.array(found4["anchor_pixel"]);point40=np.array(found40["anchor_pixel"])
        optical=np.array(anchor["pixel40_to_pixel4"])[:2,:2]
        transform=np.eye(3);transform[:2,:2]=optical;transform[:2,2]=point4-optical@point40
        global4=np.eye(3);global4[:2,2]=np.array(anchor["global4_anchor_pixel"])-point4
        physical=np.array(anchor["global4_to_anchor_microns"])
        micron4=physical@global4;micron40=micron4@transform
        w,h=found40["image_size"];corners=np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]])
        return dict(status="image_anchor_registered",anchor_file=str(anchor_path),
            anchor_sha256=hashlib.sha256(anchor_bytes).hexdigest(),
            template_sha256={k:hashlib.sha256(Path(v["template_path"]).read_bytes()).hexdigest()
                             for k,v in anchor["references"].items()},
            images={"4":str(image4) if not isinstance(image4,np.ndarray) else "in_memory_native_pixels",
                    "40":str(image40) if not isinstance(image40,np.ndarray) else "in_memory_native_pixels"},
            detections={"4":found4,"40":found40},pixel40_to_pixel4=transform.tolist(),
            pixel4_to_pixel40=np.linalg.inv(transform).tolist(),
            pixel4_to_global4=global4.tolist(),pixel40_to_global4=(global4@transform).tolist(),
            pixel4_to_anchor_microns=micron4.tolist(),pixel40_to_anchor_microns=micron40.tolist(),
            anchor_microns_to_pixel4=np.linalg.inv(micron4).tolist(),
            anchor_microns_to_pixel40=np.linalg.inv(micron40).tolist(),
            footprint40_in_pixel4=(corners@transform.T)[:,:2].tolist(),
            assumptions="Original native pixel scale and unchanged objective orientation; both images contain the saved feature.",
            physical_accuracy_verified=False,applied_to_hardware=False)

    def fit_joint_optics(self):
        """Fit shared image similarity with balanced, symmetric contour correspondences."""
        from scipy.spatial import cKDTree
        report=json.loads((self.OUTPUT/"restored_joint_source_registration.json").read_text())
        audit=json.loads((self.OUTPUT/"restored_joint_native_edge_audit.json").read_text())
        eligible=[v for v in audit["measurements"] if v["status"]=="measured" and
                  v["mean_before"]<3 and v["mean_after"]<1.5 and not v["boundary"]]
        references=sorted({v["reference4"] for v in eligible});heldout_refs=set(references[::3])
        selected=[]
        for j in references:
            group=[v for v in eligible if v["reference4"]==j]
            selected.extend(group[k] for k in np.unique(np.linspace(0,len(group)-1,min(12,len(group))).astype(int)))
        edges4={}
        for j in references:
            edge=self.edges(cv2.imread(report["frame4"][j]["path"],0),1280,False)
            y,x=np.where(edge);edges4[j]=np.column_stack([x,y]).astype(float)
        items=[];center=np.array([639.5,639.5])
        for v in selected:
            f=report["frame40"][v["index"]];edge=self.edges(cv2.imread(f["path"],0),320,True)
            y,x=np.where(edge);points=np.column_stack([(x+.5)*4-.5,(y+.5)*4-.5])-center
            if len(points)>160:points=points[np.linspace(0,len(points)-1,160).astype(int)]
            local=np.array(f["pixel40_to_reference4"]);origin=local[:2,2]+.1*center
            initial=np.clip(v["shift4px"],-12,12);target=edges4[v["reference4"]]
            distance=cKDTree(.1*points+origin+initial).query(target)[0];target=target[distance<6]
            if len(target)<20:continue
            if len(target)>200:target=target[np.linspace(0,len(target)-1,200).astype(int)]
            items.append(dict(index=v["index"],reference=v["reference4"],points=points,target=target,
                              tree=cKDTree(target),center=origin,initial=initial))
        train=[f for f in items if f["reference"] not in heldout_refs]
        heldout=[f for f in items if f["reference"] in heldout_refs]
        def optical(parameters):return .1*np.array([[1+parameters[0],-parameters[1]],[parameters[1],1+parameters[0]]])
        def pairs(f,a,shift):
            projected=f["points"]@a.T+f["center"]+shift
            d,j=f["tree"].query(projected);reverse,k=cKDTree(projected).query(f["target"])
            keep=d<6;back=reverse<6
            return np.vstack([f["points"][keep],f["points"][k[back]]]),np.vstack([f["target"][j[keep]],f["target"][back]])
        parameters=np.zeros(2);shifts=np.array([f["initial"] for f in train]);bound_hit=False
        counts={j:sum(f["reference"]==j for f in train) for j in references}
        for iteration in range(12):
            rr=[];cc=[];vv=[];rhs=[];a=optical(parameters)
            for i,f in enumerate(train):
                q,target=pairs(f,a,shifts[i]);error=np.linalg.norm(q@a.T+f["center"]+shifts[i]-target,axis=1)
                weights=np.sqrt(np.minimum(1,1/np.maximum(error,1e-6))/(len(q)*counts[f["reference"]]))
                for point,t,w in zip(q,target,weights):
                    base=t-f["center"]-.1*point;row=len(rhs)
                    rr.extend([row,row,row,row+1,row+1,row+1]);cc.extend([0,1,2+2*i,0,1,3+2*i])
                    vv.extend([.1*point[0]*w,-.1*point[1]*w,w,.1*point[1]*w,.1*point[0]*w,w])
                    rhs.extend(base*w)
            for i in range(2+2*len(train)):
                w=.05 if i<2 else .001;rr.append(len(rhs));cc.append(i);vv.append(w);rhs.append(0.)
            matrix=coo_matrix((vv,(rr,cc)),shape=(len(rhs),2+2*len(train))).tocsr()
            result=lsqr(matrix,np.array(rhs),atol=1e-9,btol=1e-9,iter_lim=2000)[0]
            bound_hit=bound_hit or abs(result[0])>=.08 or abs(result[1])>=.05
            parameters=np.clip(result[:2],[-.08,-.05],[.08,.05]);shifts=np.clip(result[2:].reshape(-1,2),-12,12)
        candidate=optical(parameters)
        def evaluate(f,a):
            shift=f["initial"].copy()
            for iteration in range(20):
                q,target=pairs(f,a,shift)
                if len(q)<20:return dict(valid=False)
                residual=target-q@a.T-f["center"];error=np.linalg.norm(residual-shift,axis=1)
                weights=np.minimum(1,1/np.maximum(error,1e-6));updated=np.clip(np.average(residual,axis=0,weights=weights),-12,12)
                if np.linalg.norm(updated-shift)<1e-4:shift=updated;break
                shift=updated
            projected=f["points"]@a.T+f["center"]+shift
            distance=np.r_[f["tree"].query(projected)[0],cKDTree(projected).query(f["target"])[0]]
            return dict(valid=True,mean_distance=float(distance.mean()),p90_distance=float(np.percentile(distance,90)),
                        shift=shift.tolist(),boundary=bool(np.any(abs(shift)>=11.99)))
        checks=[dict(index=f["index"],reference=f["reference"],baseline=evaluate(f,.1*np.eye(2)),
                     candidate=evaluate(f,candidate)) for f in heldout]
        valid=[v for v in checks if v["baseline"]["valid"] and v["candidate"]["valid"] and
               not v["baseline"]["boundary"] and not v["candidate"]["boundary"]]
        before=np.array([v["baseline"]["mean_distance"] for v in valid]);after=np.array([v["candidate"]["mean_distance"] for v in valid])
        by_ref={str(j):dict(count=sum(v["reference"]==j for v in valid),median_before=float(np.median([
            v["baseline"]["mean_distance"] for v in valid if v["reference"]==j])),median_after=float(np.median([
            v["candidate"]["mean_distance"] for v in valid if v["reference"]==j]))) for j in sorted({v["reference"] for v in valid})}
        passed=bool(len(valid)>=10 and len(by_ref)>=3 and not bound_hit and
            np.median(after)<.98*np.median(before) and np.percentile(after,90)<.98*np.percentile(before,90) and
            sum(v["median_after"]<v["median_before"] for v in by_ref.values())>=2*len(by_ref)/3)
        result=dict(method="Reference-balanced symmetric point-to-point contourICP; common similarity plus boundedper-frame translations; unchangedrestoredbaseline.",
            candidate_optical_matrix=candidate.tolist(),candidate_scale=float(np.linalg.norm(candidate[:,0])),
            candidate_rotation_degrees=float(np.degrees(np.arctan2(candidate[1,0],candidate[0,0]))),
            training_frames=len(train),heldout_frames=len(heldout),valid_heldout_frames=len(valid),heldout_references=sorted(heldout_refs),
            heldout_mean_edge_before_percentiles=np.percentile(before,[50,90,100]).tolist() if len(before) else None,
            heldout_mean_edge_after_percentiles=np.percentile(after,[50,90,100]).tolist() if len(after) else None,
            per_reference=by_ref,parameter_bound_hit=bool(bound_hit),passed_image_consistency_gate=passed,
            applied=False,checks=checks,limit="Held-out sourceimages may overlaptrainingphysicalfeatures. Edgeagreement isnot2Dtargetingaccuracy. Correspondencewindow selectedfrombaseline.")
        (self.OUTPUT/"restored_joint_optics_fit.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in result.items() if k not in ("checks","per_reference")},indent=2),flush=True)
        return result

    def restore_joint_registration(self):
        """Extend the saved joint fit without changing its reference or40x poses."""
        import hashlib
        joint_path=self.OUTPUT/"joint_stage_image_solution.json"
        joint=json.loads(joint_path.read_text())
        saved=json.loads((self.OUTPUT/"dense_source_registration.json").read_text())
        anchors={int(i):np.array(p) for i,p in joint["positions"].items()}
        frames4=saved["frame4"]; count=len(frames4)
        assert [f["index"] for f in frames4]==list(range(len(self.paths4)))
        assert all(Path(f["path"])==p for f,p in zip(frames4,self.paths4))
        b4=np.array(joint["stage_microns_to_4x_pixels"])
        stage=np.array([f["stage_xy_um"] for f in frames4])
        prior=(stage-stage[126])@b4.T+anchors[126]
        links=json.loads((self.OUTPUT/"dense_raw4_links.json").read_text())
        links+=json.loads((self.OUTPUT/"dense_raw4_crossrow_links.json").read_text())
        rr=list(range(count));cc=list(range(count));vv=[.03]*count;rhs=list(prior*.03)
        for e in links:
            row=len(rhs);rr.extend([row,row]);cc.extend([e["i"],e["j"]])
            vv.extend([-1,1]);rhs.append(e["delta"])
        mat=coo_matrix((vv,(rr,cc)),shape=(len(rhs),count)).tocsr()
        fixed=np.array(sorted(anchors));free=np.array([i for i in range(count) if i not in anchors])
        fixed_values=np.array([anchors[i] for i in fixed])
        target=np.array(rhs)-mat[:,fixed]@fixed_values
        q4=prior.copy();q4[fixed]=fixed_values
        q4[free]=np.column_stack([lsqr(mat[:,free],target[:,k],atol=1e-9,btol=1e-9,
                                      iter_lim=3000)[0] for k in (0,1)])
        center=np.array([639.5,639.5])+anchors[126]
        physical=np.eye(3);physical[:2,:2]=np.linalg.inv(b4)
        physical[:2,2]=-np.linalg.inv(b4)@center
        records4=[];records40=[];maximum_inverse_error=0.;maximum_baseline_error=0.
        for f,q in zip(frames4,q4):
            pixel=np.eye(3);pixel[:2,2]=q;microns=physical@pixel
            records4.append(dict(index=f["index"],path=f["path"],stage_xy_um=f["stage_xy_um"],
                origin_4x_pixels=q.tolist(),pixel4_to_global4=pixel.tolist(),
                pixel4_to_global_microns=microns.tolist(),global_microns_to_pixel4=np.linalg.inv(microns).tolist(),
                placement_source="fixed_saved_joint_fit" if f["index"] in anchors else "stage_and_raw4_image_links"))
        for f in joint["matches"]["frames"]:
            i=f["frame40_index"];candidate=f["candidates"][0];j=candidate["original4x_index"]
            local=np.array(candidate["original40x_to_original4x"]);pixel=local.copy()
            pixel[:2,2]+=anchors[j];microns=physical@pixel
            recovered=pixel.copy();recovered[:2,2]-=q4[j]
            maximum_baseline_error=max(maximum_baseline_error,float(np.max(abs(recovered-local))))
            maximum_inverse_error=max(maximum_inverse_error,float(np.max(abs(np.linalg.inv(microns)@microns-np.eye(3)))))
            assert Path(f["original40x_path"])==self.paths40[i]
            records40.append(dict(index=i,path=str(self.paths40[i]),stage_xy_um=saved["frame40"][i]["stage_xy_um"],
                origin_4x_pixels=pixel[:2,2].tolist(),pixel40_to_global4=pixel.tolist(),
                pixel40_to_global_microns=microns.tolist(),global_microns_to_pixel40=np.linalg.inv(microns).tolist(),
                reference4_index=j,pixel40_to_reference4=local.tolist(),reference4_to_pixel40=np.linalg.inv(local).tolist(),
                image_supported=f["status"]=="joint_image_stage",placement_source=f["status"]))
        assert sorted(f["index"] for f in records40)==list(range(len(self.paths40)))
        assert maximum_baseline_error<1e-9 and maximum_inverse_error<1e-8
        report=dict(status="restored_visual_baseline_physical_accuracy_unverified",frame4=records4,frame40=records40,
            origin_reference4_index=126,calibration=dict(M4=(-b4).tolist(),M40=saved["calibration"]["M40"],
                optical_A=(np.eye(2)*joint["optical_scale"]).tolist()),
            coordinate_definition="Common-map microns from4x calibration; original40x matrix retained separately. "
                "The preserved0.1 optical fit is not a verified physical40x calibration or permanent anchor.",
            baseline_source=str(joint_path),baseline_sha256=hashlib.sha256(joint_path.read_bytes()).hexdigest(),
            validation=dict(fixed_reference4_count=len(anchors),all40_placements_preserved=True,
                maximum_baseline_local_transform_change=maximum_baseline_error,maximum_inverse_error=maximum_inverse_error,
                original_heldout_checks=joint["heldout_validation_4x_pixels"],independent_return_accuracy_verified=False,
                maximum_required_xy_error_um=self.TARGET_ERROR_UM))
        prefix="restored_joint"
        (self.OUTPUT/(prefix+"_source_registration.json")).write_text(json.dumps(report,indent=2,allow_nan=False))
        print("Restored exact joint poses: "+json.dumps(report["validation"]),flush=True)
        self.render_dense_sources(report,prefix)

    def dense_source_registration(self, source4=None):
        """Revisit every raw source using calibrated geometry and bounded edge matching."""
        from functools import lru_cache
        from scipy.spatial import cKDTree
        import hashlib
        root = self.OUTPUT.parents[3]
        prefix="dense";dataset4="2026_11_04-16_20";source_metadata=None
        if source4 is not None:
            inventory=json.loads((self.OUTPUT/"lossless_4x_source_inventory.json").read_text())
            source_metadata=next(d for d in inventory["datasets"] if d["folder"]==source4)
            dataset4=source4;prefix="lossless_"+source4
            self.paths4=self.catalog(dataset4,source_metadata["scan_time_window"])
            m4=np.array(source_metadata["calibration"]["stage"]["M"])
        else:
            m4 = np.array(json.loads((root/"patcherbot/utils/stage_calibration_4x.json").read_text())["stage"]["M"])
        m40 = np.array(json.loads((root/"patcherbot/utils/stage_calibration_40x.json").read_text())["stage"]["M"])
        optical = m4 @ np.linalg.inv(m40); b4 = -m4
        def samples(session, paths):
            a = np.genfromtxt(self.DATA/"rig_recorder_data"/session/"movement_recording.csv",
                              delimiter=";", names=True)
            times = np.array([float(p.stem.rsplit("_",1)[1]) for p in paths])
            return np.column_stack([np.interp(times,a["timestamp"],a[k]) for k in ("st_x","st_y")])
        s4 = samples(dataset4,self.paths4)
        s40 = samples("2026_11_05-11_47",self.paths40)
        q4 = (s4-s4[126]) @ b4.T
        center = np.array([639.5,639.5])
        spatial=np.floor((s40-s40.min(0))/500).astype(int)
        heldout=(spatial[:,0]+3*spatial[:,1])%5==0
        relative40=(s40-s40[185]) @ b4.T
        previous=json.loads((self.OUTPUT/"all_source_pixel_transforms.json").read_text())
        votes=[]
        for f in previous["frames"]:
            i=f["frame40_index"]
            if heldout[i]:continue
            alternatives=[q4[c["original4x_index"]]+np.array(c["original40x_to_original4x"])[:2,2]
                +(.1*np.eye(2)-optical)@center-relative40[i] for c in f["candidates"]
                if c["status"]=="unique_edge_candidate"]
            if alternatives:votes.append(np.median(alternatives,axis=0))
        anchor=np.median(votes,axis=0)
        q40=relative40+anchor
        if source4 is not None:
            old=json.loads((self.OUTPUT/"dense_source_registration.json").read_text())
            old_ref=next(f for f in old["frame4"] if f["index"]==old["origin_reference4_index"])
            old_center=np.array(old["frame40"][185]["pixel40_to_global_microns"])@np.r_[center,1]
            offset_um=old_center[:2]+old_ref["stage_xy_um"]-s40[185]
            q40=(s40+offset_um-s4[126])@b4.T+center-optical@center
            anchor=q40[185].copy()
        # Neighbor image corrections improve each raw4 view without fitting a mosaic.
        @lru_cache(maxsize=96)
        def small4(i):
            return cv2.GaussianBlur(self.edges(cv2.imread(str(self.paths4[i]),0),512,False)
                                   .astype(np.float32),(0,0),1)
        def neighbor4(i, j=None):
            j=i+1 if j is None else j
            predicted=(q4[j]-q4[i])*.4
            dx,dy=np.rint(predicted).astype(int)
            if max(abs(dx),abs(dy))>350:
                return None
            aa=small4(i)[max(0,dy):min(512,512+dy),max(0,dx):min(512,512+dx)].copy()
            bb=small4(j)[max(0,-dy):min(512,512-dy),max(0,-dx):min(512,512-dx)].copy()
            if min(aa.sum(),bb.sum())<25:
                return None
            window=cv2.createHanningWindow((aa.shape[1],aa.shape[0]),cv2.CV_32F)
            shift,response=cv2.phaseCorrelate(aa,bb,window)
            correction=np.array([dx,dy])-np.array(shift)-predicted
            if response<.25 or np.linalg.norm(correction)>(12 if j==i+1 else 48):
                return None
            return dict(i=i,j=j,delta=((predicted+correction)/.4).tolist(),response=float(response))
        linkfile=self.OUTPUT/(prefix+"_raw4_links.json")
        if linkfile.exists():
            links=json.loads(linkfile.read_text())
        else:
            links=[]
            with ThreadPoolExecutor(self.WORKERS) as pool:
                for n,item in enumerate(pool.map(neighbor4,range(len(self.paths4)-1)),1):
                    if item is not None: links.append(item)
                    if n%1000==0: print(f"Raw4 neighbor checks {n}/{len(self.paths4)-1}",flush=True)
            linkfile.write_text(json.dumps(links))
        crossfile=self.OUTPUT/(prefix+"_raw4_crossrow_links.json")
        if crossfile.exists():
            crosslinks=json.loads(crossfile.read_text())
        else:
            stage_tree=cKDTree(q4);jobs=set()
            for i in range(0,len(q4),4):
                _,near=stage_tree.query(q4[i],k=min(250,len(q4)))
                selected=[]
                for j in np.atleast_1d(near):
                    j=int(j)
                    if abs(j-i)>80 and all(abs(j-k)>25 for k in selected):selected.append(j)
                    if len(selected)==2:break
                for j in selected:jobs.add(tuple(sorted((i,j))))
            crosslinks=[]
            with ThreadPoolExecutor(self.WORKERS) as pool:
                for n,e in enumerate(pool.map(lambda pair:neighbor4(*pair),sorted(jobs)),1):
                    if e is not None:crosslinks.append(e)
                    if n%500==0:print(f"Raw4 cross-row checks {n}/{len(jobs)}",flush=True)
            crossfile.write_text(json.dumps(crosslinks))
        links=links+crosslinks
        rr=list(range(len(q4)));cc=list(range(len(q4)));vv=[.03]*len(q4)
        yy=list(q4*.03)
        for e in links:
            row=len(yy);rr += [row,row];cc += [e["i"],e["j"]];vv += [-1,1];yy.append(e["delta"])
        mat=coo_matrix((vv,(rr,cc)),shape=(len(yy),len(q4))).tocsr()
        yy=np.array(yy)
        q4=np.column_stack([lsqr(mat,yy[:,k],atol=1e-8,btol=1e-8,iter_lim=2000)[0] for k in (0,1)])
        gauge=q4[126].copy();q4-=gauge;q40-=gauge
        seedfile=self.OUTPUT/"lossless_refined_anchor.json"
        if source4 and seedfile.exists():
            seed=json.loads(seedfile.read_text())
            if seed["source4_dataset"]==source4 and seed["accepted_for_initialization"]:
                i,j=seed["source40_index"],seed["source4_index"]
                if heldout[i]:raise RuntimeError("Anchor seed belongs to withheld region")
                if not np.allclose(optical,seed["optical_A"],atol=1e-10):
                    raise RuntimeError("Anchor seed calibration differs from this run")
                shift=q4[j]+seed["translation_xy"]-q40[i]
                q40+=shift
                print(f"Image reanchor shift in4x pixels: {shift.tolist()}",flush=True)
        tree=cKDTree(q4+center)
        @lru_cache(maxsize=128)
        def reference(i):
            e=self.edges(cv2.imread(str(self.paths4[i]),0),1280,False)
            return np.minimum(cv2.distanceTransform(1-e,cv2.DIST_L2,5),100)
        def match_frame(i):
            image=cv2.imread(str(self.paths40[i]),0)
            if image is None: raise RuntimeError(f"Unreadable40x frame {i}")
            edge=self.edges(image,320,True)
            y,x=np.where(edge>0)
            if len(x)<35:
                return dict(index=i,geometry="featureless",candidates=[])
            # Exact pixel-center coordinates through the actual optical matrix.
            native=np.column_stack([(x+.5)*4-.5,(y+.5)*4-.5])
            transformed=native @ optical.T
            small=cv2.GaussianBlur(cv2.resize(image,(320,320),interpolation=cv2.INTER_AREA),(0,0),1.2)
            gx=cv2.Sobel(small,cv2.CV_32F,1,0);gy=cv2.Sobel(small,cv2.CV_32F,0,1)
            normals=np.column_stack([gx[y,x],gy[y,x]])
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-6)
            covariance=np.linalg.eigvalsh(normals.T@normals)
            structure=float(covariance[0]/max(covariance[-1],1e-9))
            geometry="two_dimensional" if structure>.035 else "edge_only"
            lo=np.floor(transformed.min(0)).astype(int)-1
            points=transformed-lo
            size=np.ceil(points.max(0)).astype(int)+2
            template=np.zeros((size[1],size[0]),np.float32)
            rounded=np.rint(points).astype(int);template[rounded[:,1],rounded[:,0]]=1
            count=float(template.sum())
            distances,indices=tree.query(q40[i]+optical@center,k=min(100,len(q4)))
            chosen=[]
            for j in np.atleast_1d(indices):
                if all(abs(int(j)-k)>=25 for k in chosen): chosen.append(int(j))
                if len(chosen)==3:break
            candidates=[]
            for j in chosen:
                dt=reference(j);guess=q40[i]-q4[j]
                nominal=guess+lo
                search_radius=128 if source4 else 64
                start=np.maximum(0,np.floor(nominal-search_radius).astype(int))
                stop=np.minimum(1280,np.ceil(nominal+search_radius+size).astype(int))
                if np.any(stop-start<size+3):continue
                region=dt[start[1]:stop[1],start[0]:stop[0]]
                costs=cv2.matchTemplate(region,template,cv2.TM_CCORR)/count
                row,col=np.unravel_index(np.argmin(costs),costs.shape)
                fitted=(start+np.array([col,row])-lo).astype(float)
                # Subpixel minimum along both image axes; keep bounded-search failures visible.
                for axis,index in [(0,col),(1,row)]:
                    if 0<index<costs.shape[1-axis]-1:
                        values=costs[row,col-1:col+2] if axis==0 else costs[row-1:row+2,col]
                        denominator=float(values[0]-2*values[1]+values[2])
                        if denominator>1e-7:fitted[axis] += np.clip(.5*(values[0]-values[2])/denominator,-.5,.5)
                sample=transformed+fitted
                vals=cv2.remap(dt,sample[:,0].astype(np.float32).reshape(-1,1),
                    sample[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,borderValue=100).ravel()
                second=costs.copy();second[max(0,row-8):row+9,max(0,col-8):col+9]=np.inf
                alternate=float(second.min())
                margin=alternate-float(costs[row,col]) if np.isfinite(alternate) else 0.
                boundary=row==0 or col==0 or row==costs.shape[0]-1 or col==costs.shape[1]-1
                candidates.append(dict(reference4_index=j,origin=(q4[j]+fitted).tolist(),
                    mean_edge_error=float(vals.mean()),p90_edge_error=float(np.percentile(vals,90)),
                    fraction_within2=float(np.mean(vals<=2)),uniqueness_margin=margin,
                    search_boundary=bool(boundary),correction=(q4[j]+fitted-q40[i]).tolist()))
            return dict(index=i,geometry=geometry,structure_ratio=structure,candidates=candidates)
        matchfile=self.OUTPUT/(prefix+"_calibrated_edge_matches.json")
        signature=hashlib.sha256(b"dense-v2"+q4.tobytes()+q40.tobytes()+optical.tobytes()+
            "|".join(str(p) for p in self.paths4+self.paths40).encode()).hexdigest()
        # Cache contains raw-image measurements, never copied placements.
        saved=json.loads(matchfile.read_text()) if matchfile.exists() else {}
        if isinstance(saved,dict) and saved.get("signature")==signature:
            measurements=saved["measurements"]
        else:
            measurements=[]
            with ThreadPoolExecutor(self.WORKERS) as pool:
                for n,item in enumerate(pool.map(match_frame,range(len(q40))),1):
                    measurements.append(item)
                    if n%250==0:print(f"Calibrated raw40/raw4 matching {n}/{len(q40)}",flush=True)
            matchfile.write_text(json.dumps(dict(signature=signature,measurements=measurements),allow_nan=False))
        targets={}; observations=[]
        for item in measurements:
            i=item["index"]
            valid=[c for c in item["candidates"] if not c["search_boundary"] and
                   c["mean_edge_error"]<1.5 and c["fraction_within2"]>.85]
            if not valid:continue
            best=min(valid,key=lambda c:c["mean_edge_error"])
            support=[c for c in valid if np.linalg.norm(np.array(c["origin"])-best["origin"])<5]
            strong=item["geometry"]=="two_dimensional" and best["uniqueness_margin"]>.12
            observations.append((i,best,strong,len(support)))
            if strong and len(support)>=2 and not heldout[i]:targets[i]=np.array(best["origin"])
        # Fit a single stage-assisted correction field, with no free per-frame jumps.
        # Piecewise-linear time knots are regularized and image-free in withheld regions.
        knots=np.arange(0,len(q40)+80,80); design=np.zeros((len(q40),len(knots)))
        for i in range(len(q40)):
            left=min(i//80,len(knots)-2);u=(i-knots[left])/80
            design[i,left]=1-u;design[i,left+1]=u
        xx=[];yy=[];weights=[]
        for i,target in targets.items():
            xx.append(design[i]);yy.append(target-q40[i]);weights.append(1)
        for k in range(len(knots)):
            row=np.zeros(len(knots));row[k]=1;xx.append(row);yy.append([0,0]);weights.append(.15)
        for k in range(len(knots)-1):
            row=np.zeros(len(knots));row[k+1]=1;row[k]=-1
            xx.append(row);yy.append([0,0]);weights.append(.4)
        xx=np.array(xx);yy=np.array(yy);weights=np.array(weights)
        coeff=np.zeros((len(knots),2))
        for _ in range(8):
            error=np.linalg.norm(xx@coeff-yy,axis=1)
            robust=np.minimum(1,3/np.maximum(error,1e-6))*weights**2
            coeff=np.linalg.lstsq(xx*np.sqrt(robust[:,None]),yy*np.sqrt(robust[:,None]),rcond=None)[0]
        refined=q40+design@coeff
        validation_observations=[o for o in observations if heldout[o[0]] and o[2]]
        prior_errors=[float(np.linalg.norm(q40[i]-c["origin"])) for i,c,_,_ in validation_observations]
        proposed_errors=[float(np.linalg.norm(refined[i]-c["origin"])) for i,c,_,_ in validation_observations]
        support_cells={tuple(spatial[i]) for i in targets}
        correction_accepted=(len(support_cells)>=3 and len(prior_errors)>=10 and
            np.median(proposed_errors)<np.median(prior_errors) and
            np.percentile(proposed_errors,90)<np.percentile(prior_errors,90))
        if not correction_accepted:
            refined=q40.copy()
        correction_validation=dict(applied=bool(correction_accepted),
            spatial_training_cells=len(support_cells),localizable_holdouts=len(prior_errors),
            median_prior_px=float(np.median(prior_errors)) if prior_errors else None,
            median_proposed_px=float(np.median(proposed_errors)) if proposed_errors else None,
            p90_prior_px=float(np.percentile(prior_errors,90)) if prior_errors else None,
            p90_proposed_px=float(np.percentile(proposed_errors,90)) if proposed_errors else None,
            policy="Require at least3 training cells and10 localizable spatial holdouts, with both median "
                "and90th percentile improving. Passing this would still not establish absolute accuracy.")
        records=[];observation_by_id={o[0]:o for o in observations}
        for item in measurements:
            i=item["index"];transform=np.eye(3);transform[:2,:2]=optical;transform[:2,2]=refined[i]
            physical=np.eye(3);physical[:2,:2]=np.linalg.inv(b4);physical[:2,2]=-np.linalg.inv(b4)@center
            obs=observation_by_id.get(i);error=None
            if obs is not None and heldout[i] and obs[2]:error=float(np.linalg.norm(refined[i]-obs[1]["origin"]))
            reference_index=obs[1]["reference4_index"] if obs else None
            local=transform.copy()
            if reference_index is not None:local[:2,2]-=q4[reference_index]
            records.append(dict(index=i,path=str(self.paths40[i]),stage_xy_um=s40[i].tolist(),
                origin_4x_pixels=refined[i].tolist(),pixel40_to_global4=transform.tolist(),
                pixel40_to_global_microns=(physical@transform).tolist(),
                global_microns_to_pixel40=np.linalg.inv(physical@transform).tolist(),
                pixel40_to_reference4=local.tolist() if reference_index is not None else None,
                reference4_to_pixel40=np.linalg.inv(local).tolist() if reference_index is not None else None,
                image_supported=i in targets and correction_accepted,
                edge_supported=obs is not None,heldout=bool(heldout[i]),validation_error_px=error,
                local_edge_error_px=obs[1]["mean_edge_error"] if obs else None,
                geometry=item["geometry"],reference4_index=reference_index))
        physical=np.eye(3);physical[:2,:2]=np.linalg.inv(b4);physical[:2,2]=-np.linalg.inv(b4)@center
        four=[]
        for i in range(len(q4)):
            transform=np.eye(3);transform[:2,2]=q4[i]
            four.append(dict(index=i,path=str(self.paths4[i]),stage_xy_um=s4[i].tolist(),
                origin_4x_pixels=q4[i].tolist(),pixel4_to_global_microns=(physical@transform).tolist(),
                global_microns_to_pixel4=np.linalg.inv(physical@transform).tolist()))
        errors=[f["validation_error_px"] for f in records if f["validation_error_px"] is not None]
        report=dict(schema_version=1,status="provisional_coordinate_map_not_validated_for_targeting",
            source4_dataset=dataset4,source4_metadata=source_metadata["metadata_path"] if source_metadata else None,
            correction_validation=correction_validation,frame4=four,frame40=records,origin_reference4_index=126,
            calibration=dict(M4=m4.tolist(),M40=m40.tolist(),optical_A=optical.tolist()),
            coordinate_definition="Relative microns; origin is raw4 frame126 pixel center (639.5,639.5). "
                "Provisional scene anchor, not a verified etched landmark or drift-independent stage origin.",
            reference4_neighbor_constraints=len(links),image_training_frame_count=len(targets),
            withheld_frame_count=int(heldout.sum()),withheld_localizable_count=len(errors),
            withheld_unlocalizable_count=int(heldout.sum())-len(errors),
            heldout_error_px=dict(median=float(np.median(errors)),p90=float(np.percentile(errors,90))) if errors else None,
            validation_limit=f"Spatially withheld40x regions; local matches searched within{128 if source4 else 64} native4pixels of prior. "
                "Unlocalizable held-out frames retained. Edge-only matches do not establish2D position. "
                "Image residuals are consistency checks, not independent absolute metrology.",
            calibration_provenance="Existing default matrices retained. Raw4 link median residual11.14px "
                "versus29.73px for saved reference-mosaic matrix. No constant timestamp lag adopted.",
            permanent_anchor_status="Not yet verified: existing rim center has approximately150um selection uncertainty.")
        attach_accuracy_requirement(report, self.TARGET_ERROR_UM)
        rim=json.loads((root/"patcherbot/utils/scan_reference.json").read_text())
        report["approximate_target_region"]=dict(
            center_global_um=(np.array(rim["center"]["stage_xy_um"])-s4[126]).tolist(),
            radius_um=float(rim["center"]["radius_pixels"]*rim["coordinate_transform"]["microns_per_downsampled_pixel_approx"]),
            inset_um=1000.,source="patcherbot/utils/scan_reference.json",
            status="Coarse interior proposal only; old rim estimate and calibration have unvalidated uncertainty.")
        if source4:
            report["calibration_provenance"]="4x matrix from recording metadata;40x matrix fromexisting40x profile. No timestamp lag applied."
            report["initialization"]="Previous image-assisted map transferred through physical microns; initialization only"
        (self.OUTPUT/(prefix+"_source_registration.json")).write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:v for k,v in report.items() if k not in ("frame4","frame40")},indent=2),flush=True)
        build_dense_coverage(report,self.OUTPUT,prefix)
        self.render_dense_sources(report,prefix)

    def refine_lossless_anchor(self):
        """Subpixel image initialization; does not certify physical return accuracy."""
        from scipy.optimize import minimize
        seed=json.loads((self.OUTPUT/"lossless_4x_coarse_anchor_candidate.json").read_text())
        report=json.loads((self.OUTPUT/("lossless_"+seed["source4_dataset"]+"_source_registration.json")).read_text())
        optical=np.array(report["calibration"]["optical_A"])
        edge40=self.edges(cv2.imread(seed["source40_path"],0),320,True)
        edge4=self.edges(cv2.imread(seed["source4_path"],0),1280,False)
        y,x=np.where(edge40>0);native=np.column_stack([(x+.5)*4-.5,(y+.5)*4-.5])
        points=native@optical.T
        dt=np.minimum(cv2.distanceTransform(1-edge4,cv2.DIST_L2,5),100)
        def residual(translation):
            p=(points+translation).astype(np.float32)
            return cv2.remap(dt,p[:,0].reshape(-1,1),p[:,1].reshape(-1,1),cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,borderValue=100).ravel()
        initial=np.array(seed["coarse_native_translation_xy"],float)
        fit=minimize(lambda t:float(np.mean(np.minimum(residual(t),8)**2)),initial,
            method="Powell",bounds=list(zip(initial-5,initial+5)),
            options=dict(xtol=.005,ftol=1e-7,maxiter=100))
        values=residual(fit.x)
        image=cv2.imread(seed["source4_path"],0)
        overlay=np.repeat(image[:,:,None],3,axis=2)
        q=np.rint(points+fit.x).astype(int);valid=(q[:,0]>=0)&(q[:,0]<1280)&(q[:,1]>=0)&(q[:,1]<1280)
        overlay[q[valid,1],q[valid,0]]=[0,255,255]
        Image.fromarray(overlay).save(self.OUTPUT/"lossless_refined_anchor_overlay.png")
        result=dict(source4_dataset=seed["source4_dataset"],source40_index=seed["source40_index"],
            source4_index=seed["source4_index"],source40_path=seed["source40_path"],source4_path=seed["source4_path"],
            optical_A=optical.tolist(),translation_xy=fit.x.tolist(),
            mean_edge_distance_px=float(np.mean(values)),p90_edge_distance_px=float(np.percentile(values,90)),
            max_edge_distance_px=float(np.max(values)),fraction_within2_px=float(np.mean(values<=2)),
            single_frame_contour_fit_succeeded=bool(fit.success and np.mean(values)<1.5),
            accepted_for_initialization=False,
            acceptance_reason="Independent multi-frame consistency must pass before this seed is applied.",
            target_error_um=self.TARGET_ERROR_UM,physical_accuracy_verified=False,
            note="One strong-contour seed for coarse session alignment. Edge distance is not2D targeting error. "
                "All other source frames and spatially withheld checks remain required.")
        (self.OUTPUT/"lossless_refined_anchor.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        return result

    def render_dense_sources(self, report, prefix="dense"):
        """All source frames contribute; preview uses half-native4x pixels."""
        corners=np.array([[0,0,1],[1279,0,1],[0,1279,1],[1279,1279,1]])
        allcorners=[corners[:,:2]+f["origin_4x_pixels"] for f in report["frame4"]]
        allcorners += [(corners@np.array(f["pixel40_to_global4"]).T)[:,:2] for f in report["frame40"]]
        lo=np.floor(np.vstack(allcorners).min(0))-2;hi=np.ceil(np.vstack(allcorners).max(0))+3
        scale=.5; size=np.ceil((hi-lo)*scale).astype(int)
        acc=np.zeros((size[1],size[0]),np.float32);den=np.zeros_like(acc)
        def weight(side):
            t=np.minimum(np.arange(side)+1,np.arange(side,0,-1)).astype(np.float32)
            return t[:,None]*t[None,:]/t.max()**2
        def add(image,matrix):
            pts=cv2.transform(np.float32([[[0,0],[image.shape[1]-1,0],[0,image.shape[0]-1],
                [image.shape[1]-1,image.shape[0]-1]]]),matrix)[0]
            start=np.floor(pts.min(0)).astype(int);stop=np.ceil(pts.max(0)).astype(int)+1
            local=matrix.copy();local[:,2]-=start
            shape=tuple((stop-start).tolist());wt=weight(image.shape[0])
            w=cv2.warpAffine(wt,local,shape);v=cv2.warpAffine(image.astype(np.float32)*wt,local,shape)
            x,y=start;sw,sh=shape
            if x<0 or y<0 or x+sw>size[0] or y+sh>size[1]:raise RuntimeError("Dense render bounds error")
            acc[y:y+sh,x:x+sw]+=v;den[y:y+sh,x:x+sw]+=w
            return int(np.count_nonzero(w))
        counts4=[];counts40=[]
        def decode4(f):
            g=cv2.imread(f["path"],0)
            if g is None:raise RuntimeError(f["path"])
            return f,cv2.resize(g,(640,640),interpolation=cv2.INTER_AREA)
        with ThreadPoolExecutor(self.WORKERS) as pool:
            for n,(f,g) in enumerate(pool.map(decode4,report["frame4"]),1):
                m=np.array([[1.,0.,0.],[0.,1.,0.]])
                m[:,2]=(np.array(f["origin_4x_pixels"])+.5-lo)*scale
                counts4.append(add(g,m))
                if n%1000==0:print(f"Rendered raw4 {n}/{len(report['frame4'])}",flush=True)
        background=np.clip(acc/np.maximum(den,1e-9),0,255).astype(np.uint8)
        acc.fill(0);den.fill(0)
        for n,f in enumerate(report["frame40"],1):
            g=cv2.resize(cv2.imread(f["path"],0),(160,160),interpolation=cv2.INTER_AREA)
            m=np.array(f["pixel40_to_global4"])[:2].copy()
            m[:,2]+=m[:,:2]@np.array([3.5,3.5]);m[:,:2]*=8
            m[:,2]-=lo;m*=scale
            counts40.append(add(g,m))
            if n%500==0:print(f"Rendered raw40 {n}/{len(report['frame40'])}",flush=True)
        forty=np.clip(acc/np.maximum(den,1e-9),0,255).astype(np.uint8)
        mask=den>0;composite=background.copy();composite[mask]=forty[mask]
        tifffile.imwrite(self.OUTPUT/(prefix+"_all_sources_composite.tif"),composite,compression="deflate")
        preview=Image.fromarray(composite);preview.thumbnail((1800,1600));preview.save(self.OUTPUT/(prefix+"_all_sources_preview.png"))
        if prefix=="restored_joint":
            overlay=np.repeat(background[:,:,None],3,axis=2)
            overlay[mask]=np.stack([background[mask],forty[mask],forty[mask]],axis=1)
            tifffile.imwrite(self.OUTPUT/(prefix+"_overlay.tif"),overlay,photometric="rgb",compression="deflate")
            preview=Image.fromarray(overlay);preview.thumbnail((1800,1600))
            preview.save(self.OUTPUT/(prefix+"_overlay.png"))
        checks=dict(included4=len(counts4),included40=len(counts40),zero_contribution4=sum(c==0 for c in counts4),
            zero_contribution40=sum(c==0 for c in counts40),render_pixels_per_native4_pixel=scale,
            render_origin_global4=lo.tolist(),shape=size[::-1].tolist())
        (self.OUTPUT/(prefix+"_render_manifest.json")).write_text(json.dumps(checks,indent=2))
        print("DENSE COMPLETE "+json.dumps(checks),flush=True)

    def render_every_source(self, joint=None):
        """Include every40x frame, regardless of ambiguity, as requested."""
        matches = joint["matches"] if joint else json.loads((self.OUTPUT / "all_source_pixel_transforms.json").read_text())
        cached = json.loads((self.OUTPUT / "edge_continuation_links.json").read_text())
        frames = matches["frames"]
        assert sorted(f["frame40_index"] for f in frames) == list(range(len(self.paths40)))
        # Use every4x reference. Weak image links are permitted for this preview.
        positions = ({int(j):np.array(p) for j,p in joint["positions"].items()}
                     if joint else {126: np.zeros(2)})
        tree = []
        links = [e for e in cached["links4"] if "delta" in e]
        while len(positions) < len(self.ids4):
            frontier = [e for e in links if (e["i"] in positions) != (e["j"] in positions)]
            if not frontier:
                raise RuntimeError("Cannot connect all original4x views")
            e = max(frontier, key=lambda v: (2 * int(v["accepted"]) +
                    v["fraction_within2"] - .05 * v["median"]))
            delta = np.array(e["delta"]) * 2
            if e["i"] in positions:
                positions[e["j"]] = positions[e["i"]] + delta
            else:
                positions[e["i"]] = positions[e["j"]] - delta
            tree.append(e)
        low = np.floor(np.array(list(positions.values())).min(0)).astype(int)
        high = np.ceil(np.array(list(positions.values())).max(0) + 1280).astype(int)
        if joint:
            origins = np.array([positions[f["candidates"][0]["original4x_index"]] +
                np.array(f["candidates"][0]["original40x_to_original4x"])[:2,2]
                for f in frames])
            low = np.minimum(low, np.floor(origins.min(0)-2).astype(int))
            high = np.maximum(high, np.ceil(origins.max(0)+130).astype(int))
        width, height = high - low
        acc4 = np.zeros((height, width), np.float32); den4 = np.zeros_like(acc4)
        acc40 = np.zeros_like(acc4); den40 = np.zeros_like(acc4)
        def weights(side):
            x = np.minimum(np.arange(side) + 1, np.arange(side, 0, -1)).astype(np.float32)
            return (x[:, None] * x[None, :]) / x.max() ** 2
        w4 = weights(1280)
        for j, position in positions.items():
            image = cv2.imread(str(self.paths4[j]), 0)
            if image is None:
                raise RuntimeError(f"Cannot read original4x frame {j}")
            x, y = np.rint(position - low).astype(int)
            acc4[y:y+1280, x:x+1280] += image * w4
            den4[y:y+1280, x:x+1280] += w4
        available = [f for f in frames if f["candidates"]]
        available_times = np.array([float(Path(f["original40x_path"]).stem.rsplit("_", 1)[1])
                                    for f in available])
        side = round(1280 * matches["shared_scale_native40x_to_native4x"])
        wt = weights(side)
        manifest = []
        def decode(f):
            image = cv2.imread(f["original40x_path"], 0)
            if image is None:
                raise RuntimeError(f"Cannot read40x frame {f['frame40_index']}")
            return f, cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
        with ThreadPoolExecutor(self.WORKERS) as pool:
            for n, (f, image) in enumerate(pool.map(decode, frames), 1):
                missing_candidate = not f["candidates"]
                estimated = missing_candidate or (joint is not None and f["status"] != "joint_image_stage")
                if missing_candidate:
                    timestamp = float(Path(f["original40x_path"]).stem.rsplit("_", 1)[1])
                    nearest = available[int(np.argmin(np.abs(available_times - timestamp)))]
                    chosen = nearest["candidates"][0]
                    placement_source = nearest["frame40_index"]
                else:
                    chosen = f["candidates"][0]
                    placement_source = f["frame40_index"]
                j = chosen["original4x_index"]
                native = np.array(chosen["original40x_to_original4x"])
                global_native = native.copy()
                global_native[:2, 2] += positions[j] - low
                resized = global_native[:2].copy()
                resized[:, 2] += resized[:, :2] @ np.full(2, .5 / (side / 1280) - .5)
                resized[:, :2] /= side / 1280
                corners = cv2.transform(np.float32([[[0, 0], [side-1, 0],
                    [0, side-1], [side-1, side-1]]]), resized)[0]
                start = np.floor(corners.min(0)).astype(int)
                stop = np.ceil(corners.max(0)).astype(int) + 1
                if np.any(start < 0) or stop[0] > width or stop[1] > height:
                    raise RuntimeError(f"Would clip40x frame {f['frame40_index']}")
                local = resized.copy(); local[:, 2] -= start
                size = tuple((stop - start).tolist())
                warped = cv2.warpAffine(image.astype(np.float32) * wt, local, size)
                weight = cv2.warpAffine(wt, local, size)
                x, y = start; w, h = size
                acc40[y:y+h, x:x+w] += warped
                den40[y:y+h, x:x+w] += weight
                pixels = int(np.count_nonzero(weight))
                assert pixels > 0
                manifest.append(dict(frame40_index=f["frame40_index"],
                    original40x_path=f["original40x_path"], original4x_index=j,
                    original4x_path=str(self.paths4[j]), match_status=f["status"],
                    estimated_placement=estimated, placement_source_frame40=placement_source,
                    original40x_to_original4x=native.tolist(),
                    original4x_to_original40x=np.linalg.inv(native).tolist(),
                    original40x_to_output_pixels=global_native.tolist(),
                    contributing_output_pixels=pixels))
                if n % 500 == 0:
                    print(f"Included {n}/{len(frames)}40x images", flush=True)
        assert len(manifest) == len(self.paths40)
        assert len({m["frame40_index"] for m in manifest}) == len(self.paths40)
        background = np.clip(acc4 / np.maximum(den4, 1e-6), 0, 255).astype(np.uint8)
        forty = np.clip(acc40 / np.maximum(den40, 1e-6), 0, 255).astype(np.uint8)
        covered = den40 > 0
        composite = background.copy(); composite[covered] = forty[covered]
        color = np.repeat(background[:, :, None], 3, 2)
        color[covered] = np.stack([background[covered], forty[covered], forty[covered]], 1)
        prefix = "joint_stage_image_all2147" if joint else "all_2147_40x_on_4x"
        for name, data in [(prefix, composite), (prefix+"_overlay", color)]:
            tifffile.imwrite(self.OUTPUT / (name + ".tif"), data,
                photometric="rgb" if data.ndim == 3 else "minisblack", compression="deflate")
            preview = Image.fromarray(data); preview.thumbnail((1800, 1600))
            preview.save(self.OUTPUT / (name + ".png"))
        report = dict(rendered_original40x_count=len(manifest),
            total_original40x_count=len(self.paths40), excluded_original40x_count=0,
            original4x_count=len(positions), output_shape=[int(height), int(width)],
            estimated_placement_count=sum(m["estimated_placement"] for m in manifest),
            placement_policy=("Joint robust image constraints and relative stage trajectory; every frame included."
                if joint else "Best edge candidate, or temporally nearest candidate."),
            reference_layout_policy=("Stage priors and accepted original-image links" if joint
                else "Image-only spanning tree including weak links"),
            reference_positions={str(j): v.tolist() for j, v in positions.items()},
            reference_links=tree, frames=manifest)
        (self.OUTPUT / (prefix+"_manifest.json")).write_text(
            json.dumps(report, indent=2, allow_nan=False))
        print(f"COMPLETE: {len(manifest)} included, 0excluded; "
              f"{report['estimated_placement_count']}estimatedplacements", flush=True)


def attach_accuracy_requirement(report, maximum_error_um=2.0):
    """Record image consistency separately from actual physical return accuracy."""
    inverse=np.linalg.inv(-np.asarray(report["calibration"]["M4"],float))
    # This calibration is isotropic; the norm is conservative if a future one is not.
    micron_bound_per_pixel=float(np.linalg.norm(inverse,2))
    measurements=[f["validation_error_px"]*micron_bound_per_pixel for f in report["frame40"]
                  if f["heldout"] and f["validation_error_px"] is not None]
    report["accuracy_requirement"]=dict(maximum_xy_error_um=float(maximum_error_um),
        acceptance_measurement="Independent physical return visits across the intended working area",
        independent_return_visits_verified=False,target_met=False,
        native4_pixel_error_equivalent=float(maximum_error_um/micron_bound_per_pixel),
        heldout_image_checks=len(measurements),
        heldout_image_checks_within_target=sum(e<=maximum_error_um for e in measurements),
        maximum_observed_image_disagreement_um=max(measurements) if measurements else None,
        note="Image correspondence residuals do not verify physical return-position accuracy. "
             "Unlocalizable frames and unobserved regions remain unvalidated.")
    return report


def build_dense_coverage(report, output, prefix="dense"):
    """Save offline footprint coverage and an operator-reviewed acquisition batch.

    Coordinates are relative to the center of reference frame 126. Coverage is
    sampled at 50 um and summarized in 500 um cells; it is not a tissue mask.
    Held-out image residuals measure internal agreement, not physical accuracy.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch, Circle

    output = Path(output)
    if not output.is_dir():
        raise ValueError("The existing result directory must be supplied")
    frames4, frames40 = report["frame4"], report["frame40"]
    if not frames4:
        raise ValueError("Coverage requires original 4x frames")
    inverse = np.linalg.inv(-np.asarray(report["calibration"]["M4"], float))
    camera_center = np.array([639.5, 639.5])
    corners = np.array([[-.5, -.5], [1279.5, -.5],
                        [1279.5, 1279.5], [-.5, 1279.5]])
    reference = next(f for f in frames4
                     if f["index"] == report["origin_reference4_index"])
    stage_anchor = np.asarray(reference["stage_xy_um"], float)

    def physical(points):
        return (np.asarray(points) - camera_center) @ inverse.T

    polygons4 = [physical(corners + np.asarray(f["origin_4x_pixels"]))
                 for f in frames4]
    polygons40 = []
    for frame in frames40:
        matrix = np.asarray(frame["pixel40_to_global4"], float)
        homogeneous = np.column_stack([corners, np.ones(4)]) @ matrix.T
        polygons40.append(physical(homogeneous[:, :2] / homogeneous[:, 2:]))
    points = np.concatenate(polygons4 + polygons40)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite frame footprints cannot define coverage")
    cell_um, sample_um = 500.0, 50.0
    lower = np.floor(points.min(0) / cell_um) * cell_um
    upper = np.ceil(points.max(0) / cell_um) * cell_um
    nx, ny = np.maximum(1, np.rint((upper - lower) / cell_um).astype(int))
    if nx * ny > 100000:
        raise ValueError("Implausible coverage extent; inspect registration coordinates")
    samples = int(cell_um / sample_um)
    shape = (int(ny * samples), int(nx * samples))
    coverage4 = np.zeros(shape, np.uint8)
    coverage40 = np.zeros(shape, np.uint8)
    measured = np.zeros(shape, np.uint8)
    edge_only = np.zeros(shape, np.uint8)
    measured2d = np.zeros(shape, np.uint8)
    checked2d = np.zeros(shape, np.uint8)
    residual_cells = {}
    frame_counts4 = np.zeros((ny, nx), int)
    frame_counts40 = np.zeros((ny, nx), int)

    def stamp(polygon, masks, counts):
        # Fixed-point polygon vertices preserve subpixel footprint coordinates.
        pixel = (polygon - lower) / sample_um - .5
        start = np.maximum(0, np.floor(pixel.min(0)).astype(int) - 1)
        stop = np.minimum([shape[1], shape[0]], np.ceil(pixel.max(0)).astype(int) + 2)
        if np.any(stop <= start):
            return []
        patch = np.zeros(tuple((stop - start)[::-1]), np.uint8)
        fixed = np.rint((pixel - start) * 256).astype(np.int32)
        cv2.fillConvexPoly(patch, fixed, 1, shift=8)
        x0, y0 = start; x1, y1 = stop
        for mask in masks:
            np.maximum(mask[y0:y1, x0:x1], patch, out=mask[y0:y1, x0:x1])
        yy, xx = np.nonzero(patch)
        ids = np.unique(((yy + y0) // samples) * nx + (xx + x0) // samples)
        counts.flat[ids] += 1
        return ids

    for polygon in polygons4:
        stamp(polygon, [coverage4], frame_counts4)
    for frame, polygon in zip(frames40, polygons40):
        masks = [coverage40]
        supported = bool(frame["image_supported"] or frame.get("edge_supported",False))
        geometry = frame["geometry"]
        error = frame.get("validation_error_px")
        finite_error = error is not None and np.isfinite(error)
        checked = geometry == "two_dimensional" and frame["heldout"] and finite_error
        if supported:
            masks.append(measured)
            if geometry == "edge_only":
                masks.append(edge_only)
            elif geometry == "two_dimensional":
                masks.append(measured2d)
        if checked:
            masks.append(checked2d)
        ids = stamp(polygon, masks, frame_counts40)
        if frame["heldout"] and finite_error:
            for index in ids:
                residual_cells.setdefault(int(index), []).append(float(error))

    def fractions(mask):
        return mask.reshape(ny, samples, nx, samples).mean(axis=(1, 3))

    f4, f40, fm, fe, f2, fc = map(fractions,
                              [coverage4, coverage40, measured, edge_only, measured2d, checked2d])
    status = np.zeros((ny, nx), np.uint8)
    status[f4 > 0] = 1
    status[(f4 > 0) & (f40 > 0)] = 2
    status[(f4 > 0) & (fe > 0)] = 3
    status[(f4 > 0) & (f2 > 0)] = 4
    status[(f4 > 0) & (fc > 0)] = 5
    labels = ["No 4x footprint", "4x only: missing 40x", "40x placement unverified",
              "Edge response; position ambiguous", "2D candidate; accuracy unverified", "2D checked; inspect error"]
    colors = ["#eeeeee", "#d9b44a", "#bf727b", "#8d82b5", "#579ca4", "#de7736"]
    residual = np.full((ny, nx), np.nan)
    micron_bound_per_pixel=float(np.linalg.norm(inverse,2))
    for index, errors in residual_cells.items():
        residual.flat[index] = float(np.max(errors)*micron_bound_per_pixel)
    cells, candidates = [], []
    for row in range(ny):
        for col in range(nx):
            if not f4[row, col]:
                continue
            center = lower + (np.array([col, row]) + .5) * cell_um
            item = dict(row=row, column=col, global_xy_um=center.tolist(),
                coverage4_fraction=float(f4[row, col]), coverage40_fraction=float(f40[row, col]),
                measured_fraction=float(fm[row, col]), edge_only_fraction=float(fe[row, col]),
                heldout_2d_fraction=float(fc[row, col]), status=labels[status[row, col]],
                source4_count=int(frame_counts4[row, col]), source40_count=int(frame_counts40[row, col]),
                maximum_heldout_image_disagreement_um=(float(residual[row, col])
                                         if np.isfinite(residual[row, col]) else None))
            cells.append(item)
            # Partial cells remain eligible: a 40x field is smaller than a cell.
            if fc[row, col] < f4[row, col] * .8:
                missing = max(0., f4[row, col] - f40[row, col])
                weak = max(0., min(f4[row, col], f40[row, col]) - fc[row, col])
                item["priority"] = float(2 * missing + weak)
                sy, sx = row * samples, col * samples
                eligible_samples = ((coverage4[sy:sy + samples, sx:sx + samples] > 0) &
                                    (checked2d[sy:sy + samples, sx:sx + samples] == 0))
                yy, xx = np.nonzero(eligible_samples)
                if not len(xx):
                    continue
                closest = np.argmin((xx - (samples - 1) / 2) ** 2 +
                                    (yy - (samples - 1) / 2) ** 2)
                item["target_xy_um"] = (lower +
                    (np.array([sx + xx[closest], sy + yy[closest]]) + .5) * sample_um).tolist()
                region=report.get("approximate_target_region")
                if region is not None and np.linalg.norm(np.array(item["target_xy_um"])-
                    region["center_global_um"]) > region["radius_um"]-region["inset_um"]:
                    continue
                candidates.append(item)
    candidates.sort(key=lambda c: (-c["priority"], c["row"], c["column"]))
    targets = []
    while candidates and len(targets) < 20:
        if targets:
            eligible = [(c, min(np.linalg.norm(np.asarray(c["target_xy_um"]) - t["global_xy_um"])
                               for t in targets)) for c in candidates]
            eligible = [(c, distance) for c, distance in eligible if distance >= 1000.]
            if not eligible:
                break
            chosen = max(eligible, key=lambda pair: pair[0]["priority"] *
                         (1 + min(pair[1] / 2000., 2)))[0]
        else:
            chosen = candidates[0]
        candidates.remove(chosen)
        xy = np.asarray(chosen["target_xy_um"])
        targets.append(dict(rank=len(targets) + 1, global_xy_um=xy.tolist(),
            recorded4x_stage_xy_um=(stage_anchor + xy).tolist(), priority=chosen["priority"],
            reason=("Missing 40x coverage" if chosen["coverage40_fraction"] == 0
                    else "Partial or insufficiently checked 40x coverage"),
            operator_review_required=True))
    extent = [lower[0], upper[0], lower[1], upper[1]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    cmap = ListedColormap(colors)
    axes[0].imshow(status, origin="lower", extent=extent, cmap=cmap,
                   norm=BoundaryNorm(np.arange(-.5, 6.5), cmap.N), interpolation="nearest")
    axes[0].set_title("Strongest available evidence per 500 µm cell")
    axes[0].legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, labels)],
                   fontsize=7, loc="lower left")
    area = np.ma.masked_where(f4 == 0, f40)
    artist = axes[1].imshow(area, origin="lower", extent=extent, vmin=0, vmax=1,
                            cmap="viridis", interpolation="nearest")
    fig.colorbar(artist, ax=axes[1], label="Cell area covered by 40x footprints")
    axes[1].set_title("40x coverage and proposed next acquisition batch")
    region=report.get("approximate_target_region")
    if region is not None:
        axes[1].add_patch(Circle(region["center_global_um"],region["radius_um"]-region["inset_um"],
            fill=False,edgecolor="white",linestyle="--",linewidth=1))
    for target in targets:
        x, y = target["global_xy_um"]
        axes[1].plot(x, y, "o", color="white", markersize=5, markeredgecolor="black")
        axes[1].annotate(str(target["rank"]), (x, y), xytext=(4, 3),
                         textcoords="offset points", fontsize=7)
    artist = axes[2].imshow(np.ma.masked_invalid(residual), origin="lower", extent=extent,
                            cmap="magma", interpolation="nearest")
    fig.colorbar(artist, ax=axes[2], label="Maximum held-out image disagreement (µm)")
    axes[2].set_title("2 µm target; blank = unmeasured")
    for axis in axes:
        axis.set_xlabel("Relative global X (µm)"); axis.set_ylabel("Relative global Y (µm)")
        axis.set_aspect("equal")
    fig.suptitle("Footprint coverage is not a tissue/basin mask. Held-out agreement is not ground truth.", fontsize=11)
    fig.savefig(output / (prefix+"_coverage_accuracy.png"), dpi=160)
    plt.close(fig)
    result = dict(cell_size_um=cell_um, footprint_sample_um=sample_um,
        origin_reference4_index=report["origin_reference4_index"],
        coordinate_definition="inv(-M4) @ (global4_pixels - [639.5,639.5])",
        recorded_stage_anchor_xy_um=stage_anchor.tolist(), source4_count=len(frames4),
        source40_count=len(frames40), extent_um=extent, grid_shape=[int(ny), int(nx)],
        approximate_target_region=report.get("approximate_target_region"),
        accuracy_requirement=report.get("accuracy_requirement"),
        notes=["Footprints use the supplied transforms; inferred placements are not measurements.",
               "Evidence categories report the strongest available evidence, not whole-cell validation.",
               "Raster coverage is approximate at 50 um sampling; overlaps are unioned.",
               "Held-out residuals are internal image agreement, not physical accuracy or independent ground truth.",
               "Proposed positions are an adaptive first batch of at most 20, spaced at least 1 mm apart; no completeness guarantee.",
               "Tissue, coverslip and collision masks are unknown. Operator review and experiment re-anchoring are required before any movement.",
               "Recorded stage targets refer to the 4x recording coordinate frame; no objective-offset correction is implied."],
        cells=cells, targeted_acquisition=targets)
    (output / (prefix+"_coverage_accuracy.json")).write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    target_name="targeted_acquisition.csv" if prefix=="dense" else prefix+"_targeted_acquisition.csv"
    with (output / target_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "global_x_um", "global_y_um", "recorded4x_stage_x_um",
                         "recorded4x_stage_y_um", "priority", "reason", "operator_review_required"])
        for target in targets:
            writer.writerow([target["rank"], *target["global_xy_um"],
                             *target["recorded4x_stage_xy_um"], target["priority"], target["reason"], True])
    return result


if __name__ == "__main__":
    EdgeRegistration().restore_joint_registration()
