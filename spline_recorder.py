"""
Spline recording utilities for MPPI debugging.

Usage in your main MPPI script:

    from spline_recorder import SplineRecorder
    
    # After creating mppi controller:
    recorder = SplineRecorder(out_path="spline_log.npz",
                              record_every=10,        # log every 10 control steps
                              n_samples_to_keep=5,    # keep this many samples per step
                              max_steps=200)          # stop logging after this many records

    # Inside your main loop, AFTER mppi.control_step(...) but BEFORE applying ctrl:
    recorder.record(
        step=step,
        live_qpos=live_data.qpos.copy(),
        live_qvel=live_data.qvel.copy(),
        tau_best_q=mppi.tau_best_q,
        tau_best_v=mppi.tau_best_v,
        tau0_q=mppi.tau0_q,
        tau0_v=mppi.tau0_v,
        # Optional: pass extra debug info if you have it:
        last_iter_samples_q=getattr(mppi, '_dbg_last_iter_sq', None),
        last_iter_samples_v=getattr(mppi, '_dbg_last_iter_sv', None),
        last_iter_costs=getattr(mppi, '_dbg_last_iter_costs', None),
    )

    # After the loop ends:
    recorder.save()

Then run `python plot_splines.py spline_log.npz` to view the plots.
"""

import numpy as np


class SplineRecorder:
    def __init__(self, out_path, record_every=10, n_samples_to_keep=5, max_steps=200):
        self.out_path = out_path
        self.record_every = record_every
        self.n_samples_to_keep = n_samples_to_keep
        self.max_steps = max_steps
        self.records = []

    def record(self, step, live_qpos, live_qvel, tau_best_q, tau_best_v,
               tau0_q, tau0_v,
               last_iter_samples_q=None, last_iter_samples_v=None,
               last_iter_costs=None):
        if len(self.records) >= self.max_steps:
            return
        if step % self.record_every != 0:
            return

        rec = {
            'step': step,
            'live_qpos': live_qpos.copy(),
            'live_qvel': live_qvel.copy(),
            'tau_best_q': tau_best_q.copy(),
            'tau_best_v': tau_best_v.copy(),
            'tau0_q': tau0_q.copy(),
            'tau0_v': tau0_v.copy(),
        }

        # Optionally keep a few representative samples — the best, worst, median by cost
        if last_iter_samples_q is not None and last_iter_costs is not None:
            costs = np.asarray(last_iter_costs)
            finite = costs < 1e7
            if finite.any():
                order = np.argsort(costs)
                n = min(self.n_samples_to_keep, len(order))
                # Take: best, second-best, median, worst-finite, and unperturbed sample (idx 0)
                pick = [order[0]]
                if n >= 2: pick.append(order[1])
                if n >= 3:
                    median_idx = order[len(order) // 2]
                    if median_idx not in pick: pick.append(median_idx)
                if n >= 4:
                    finite_order = order[finite[order]]
                    if len(finite_order) > 0 and finite_order[-1] not in pick:
                        pick.append(finite_order[-1])
                if n >= 5 and 0 not in pick:
                    pick.append(0)
                rec['sample_q'] = last_iter_samples_q[pick].copy()
                rec['sample_v'] = last_iter_samples_v[pick].copy()
                rec['sample_cost'] = costs[pick].copy()
                rec['sample_idx'] = np.array(pick)

        self.records.append(rec)

    def save(self):
        """Save all recorded data to npz. Atomic: writes to a tmp file
        first and renames at the end so a partial file is never left on
        disk if the process dies during save."""
        if not self.records:
            print(f"SplineRecorder: nothing to save")
            return
        # Convert list-of-dicts into dict-of-arrays for npz
        keys = self.records[0].keys()
        out = {}
        for k in keys:
            try:
                out[k] = np.stack([r[k] for r in self.records if k in r])
            except (ValueError, KeyError):
                pass
        # Also save metadata
        out['_n_records'] = np.array(len(self.records))
        out['_record_every'] = np.array(self.record_every)
        # Atomic write: tmp + rename.
        # np.savez_compressed auto-appends '.npz' to the path, so we strip
        # that suffix from out_path before appending .tmp, then compute the
        # final paths it will actually write to.
        import os
        base = self.out_path
        if base.endswith('.npz'):
            base = base[:-4]
        tmp_path_arg = base + ".tmp"          # what we pass to savez
        tmp_path_actual = tmp_path_arg + ".npz"  # what gets written
        final_path = base + ".npz"
        np.savez_compressed(tmp_path_arg, **out)
        os.replace(tmp_path_actual, final_path)
        print(f"SplineRecorder: saved {len(self.records)} records to {final_path}")