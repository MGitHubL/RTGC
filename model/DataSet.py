from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from torch.utils.data import Dataset


class TGCDataSet(Dataset):

    def __init__(
        self,
        file_path: str,
        neg_size: int,
        hist_len: int,
        feature_path: str,
        directed: bool,
        transform=None,
        neg_sampling_power: float = 0.75,
        neg_table_size: int = int(1e7),
    ) -> None:
        self.neg_size = int(neg_size)
        self.hist_len = int(hist_len)
        self.directed = directed
        self.transform = transform
        self.feature_path = feature_path

        self.max_d_time = float("-inf")
        self.NEG_SAMPLING_POWER = float(neg_sampling_power)
        self.neg_table_size = int(neg_table_size)

        temp_hist: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

        node_set = set()
        edge_num = 0
        max_node_id = -1

        with open(file_path, "r") as infile:
            for line_no, line in enumerate(infile, start=1):
                parts = line.split()
                if len(parts) < 3:
                    continue

                s_node = int(parts[0])
                t_node = int(parts[1])
                d_time = float(parts[2])

                edge_num += 1
                node_set.add(s_node)
                node_set.add(t_node)
                max_node_id = max(max_node_id, s_node, t_node)

                temp_hist[s_node].append((t_node, d_time))
                if not directed:
                    temp_hist[t_node].append((s_node, d_time))

                if d_time > self.max_d_time:
                    self.max_d_time = d_time

        if len(node_set) == 0:
            raise ValueError("Empty graph: no valid nodes found in input file.")

        self.edge_num = edge_num
        self.node_dim = len(node_set)

        if max_node_id != self.node_dim - 1 or min(node_set) != 0:
            raise ValueError(
                f"Node IDs are not continuous from 0. "
                f"min_id={min(node_set)}, max_id={max_node_id}, node_dim={self.node_dim}"
            )

        self.degrees = np.zeros(self.node_dim, dtype=np.int64)
        for s_node, hist in temp_hist.items():
            self.degrees[s_node] += len(hist)

        self.node2hist: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.data_size = 0

        for s_node, hist in temp_hist.items():

            hist.sort(key=lambda x: x[1])
            hist_nodes = np.fromiter(
                (x[0] for x in hist), dtype=np.int64, count=len(hist)
            )
            hist_times = np.fromiter(
                (x[1] for x in hist), dtype=np.float32, count=len(hist)
            )
            self.node2hist[s_node] = (hist_nodes, hist_times)
            self.data_size += len(hist)

        self.idx2source_id = np.empty(self.data_size, dtype=np.int32)
        self.idx2target_id = np.empty(self.data_size, dtype=np.int32)

        idx = 0
        for s_node, (hist_nodes, _) in self.node2hist.items():
            n = hist_nodes.shape[0]
            self.idx2source_id[idx : idx + n] = s_node
            self.idx2target_id[idx : idx + n] = np.arange(n, dtype=np.int32)
            idx += n

        self.neg_table = self._init_neg_table()

    def get_node_dim(self) -> int:
        return self.node_dim

    def get_edge_num(self) -> int:
        return self.edge_num

    def get_max_d_time(self) -> float:
        return self.max_d_time

    def get_feature(self) -> np.ndarray:
        feat_dim = None
        feature = None

        with open(self.feature_path, "r") as reader:
            _ = reader.readline()
            for line in reader:
                arr = np.fromstring(line.strip(), dtype=np.float32, sep=" ")
                if arr.size < 2:
                    continue

                node_id = int(arr[0])
                emb = arr[1:]

                if node_id < 0 or node_id >= self.node_dim:
                    raise ValueError(
                        f"Feature node_id {node_id} out of range [0, {self.node_dim - 1}]"
                    )

                if feat_dim is None:
                    feat_dim = emb.shape[0]
                    feature = np.zeros((self.node_dim, feat_dim), dtype=np.float32)
                elif emb.shape[0] != feat_dim:
                    raise ValueError(
                        f"Inconsistent feature dim at node {node_id}: "
                        f"{emb.shape[0]} vs expected {feat_dim}"
                    )

                feature[node_id] = emb

        if feature is None:
            raise ValueError("No valid feature rows found.")

        return feature

    def _init_neg_table(self) -> np.ndarray:
        deg = self.degrees.astype(np.float64)
        weights = np.power(deg, self.NEG_SAMPLING_POWER)

        w_sum = weights.sum()
        if w_sum <= 0:

            return np.random.randint(
                0, self.node_dim, size=self.neg_table_size, dtype=np.int32
            )

        probs = weights / w_sum
        cdf = np.cumsum(probs)

        u = np.linspace(
            0.0, 1.0, num=self.neg_table_size, endpoint=False, dtype=np.float64
        )
        table = np.searchsorted(cdf, u, side="right").astype(np.int32)
        return table

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s_node = int(self.idx2source_id[idx])
        t_idx = int(self.idx2target_id[idx])

        hist_nodes_all, hist_times_all = self.node2hist[s_node]

        t_node = int(hist_nodes_all[t_idx])
        t_time = float(hist_times_all[t_idx])

        start = t_idx - self.hist_len
        if start < 0:
            start = 0

        h_nodes = hist_nodes_all[start:t_idx]
        h_times = hist_times_all[start:t_idx]
        h_len = h_nodes.shape[0]

        np_h_nodes = np.zeros(self.hist_len, dtype=np.int64)
        np_h_times = np.zeros(self.hist_len, dtype=np.float32)
        np_h_masks = np.zeros(self.hist_len, dtype=np.float32)

        if h_len > 0:
            np_h_nodes[:h_len] = h_nodes
            np_h_times[:h_len] = h_times
            np_h_masks[:h_len] = 1.0

        neg_nodes = self.negative_sampling()

        sample = {
            "source_node": s_node,
            "target_node": t_node,
            "target_time": t_time,
            "history_nodes": np_h_nodes,
            "history_times": np_h_times,
            "history_masks": np_h_masks,
            "neg_nodes": neg_nodes,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self) -> np.ndarray:
        rand_idx = np.random.randint(
            0, self.neg_table_size, size=self.neg_size, dtype=np.int64
        )
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes
