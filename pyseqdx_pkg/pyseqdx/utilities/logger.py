import pandas as pd
import numpy as np
import torch

import json
import pickle
import os
import copy

from typing import Optional


class MetricLogger:
    def __init__(
        self,
        keys: list,
        context: Optional[dict] = None,
        allow_unknown_keys=True,
        tempfile_path=None,  # temp file to save during logging
        overwrite_tempfile=False,  # whether overwrite tempfile if already exists
    ):
        self._keys = set(keys)
        if context:
            self._context = context
        else:
            self._context = {}
        self._allow_unknown_keys = allow_unknown_keys
        self._records = []
        self._jsonlpart_path = tempfile_path

        if self._jsonlpart_path:
            if os.path.exists(self._jsonlpart_path) and not overwrite_tempfile:
                raise FileExistsError(
                    f"jsonl_path already exists: {self._jsonlpart_path}"
                )
            os.makedirs(os.path.dirname(self._jsonlpart_path), exist_ok=True)
            open(self._jsonlpart_path, "w").close()  # create empty file
            with open(self._jsonlpart_path, "w") as f:
                f.write(json.dumps({"__context__": self._context}) + "\n")

    def __repr__(self):
        # print(tracker) method
        try:
            df = self.to_dataframe().map(self._safe_scalar)
            return f"<MetricLogger with {len(self._records)} records>\n{df}"
        except Exception as e:
            return f"<MetricLogger (error displaying dataframe): {e}>"

    def add_context(self, context: dict, overwrite: bool = False):
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        if not overwrite:
            conflict = set(context) & set(self._context)
            if conflict:
                raise ValueError(f"Context keys already exist: {conflict}")
        self._context.update(context)

    def log(self, **kwargs):
        self.log_dict(kwargs)

    def log_dict(self, record: dict):
        if not self._allow_unknown_keys:
            unknown_keys = set(record) - self._keys
            if unknown_keys:
                raise ValueError(f"Unknown keys: {unknown_keys}")

        # freeze before recording
        freeze_record = {
            k: v.clone().detach() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
            for k, v in record.items()
        }

        self._records.append(freeze_record)
        if self._jsonlpart_path:
            self._append_jsonl(freeze_record)

    def _append_jsonl(self, record):
        # streaming to tempfile
        safe_record = {k: self._json_safe(v) for k, v in record.items()}
        json_line = json.dumps(safe_record)
        with open(self._jsonlpart_path, "a") as f:
            f.write(json_line + "\n")

    def to_dataframe(self):
        # pop to pd.dataframe
        rows = []
        for record in self._records:
            row = {}
            for k, v in record.items():
                if isinstance(v, pd.DataFrame):
                    row[k] = v.copy()  # store as object reference
                # elif isinstance(v, torch.Tensor):
                #     row[k] = v.numpy(force=True)
                else:
                    row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows)

        # Add context tags as new columns
        for k, v in self._context.items():
            use_k = f"context__{k}"
            df[use_k] = v

        return df

    def to_csv(self, filepath):
        # save to csv, only scalar, list, and tuple values will be saved
        # tensor, np.arrays, pd.dataframe will provide shape
        df = self.to_dataframe()
        df = df.map(self._safe_scalar)
        df.to_csv(filepath, index=False)

    def to_pickle(self, filepath):
        # pickle
        with open(filepath, "wb") as f:
            pickle.dump({"records": self._records, "context": self._context}, f)

    @classmethod
    def from_pickle(cls, filepath):
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        records = data.get("records", [])
        context = data.get("context", {})

        all_keys = set()
        for r in records:
            all_keys.update(r.keys())

        tracker = cls(keys=list(all_keys), context=context, allow_unknown_keys=True)
        tracker._records = records
        return tracker

    def to_json(self, filepath=None, flush_tempfile=True):
        # pop/save to json object/file

        # apply _json_safe to each row (element (that is dict itself) in list)
        def recursive_safe(obj):
            return {k: self._json_safe(v) for k, v in obj.items()}

        safe_records = [recursive_safe(row) for row in self._records]

        json_obj = {"context": self._context, "records": safe_records}
        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(json_obj, f, indent=2)
            # Flush jsonl cache
            if (
                flush_tempfile
                and self._jsonlpart_path
                and os.path.exists(self._jsonlpart_path)
            ):
                os.remove(self._jsonlpart_path)
        else:
            return json_obj

    @classmethod
    def from_json(cls, json_obj=None, filepath=None):
        if json_obj is None:
            with open(filepath, "r") as f:
                raw = json.load(f)
        else:
            raw = json_obj

        context = raw.get("context", {})
        records = raw.get("records", [])

        if not records:
            raise ValueError("No records found in JSON.")

        all_keys = set()
        for r in records:
            all_keys.update(r.keys())

        tracker = cls(keys=list(all_keys), context=context, allow_unknown_keys=True)
        tracker._records = records
        return tracker

    @classmethod
    def from_jsonl(cls, filepath):
        import json

        with open(filepath, "r") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("Empty JSONL file.")

        # Parse context from first line
        first = json.loads(lines[0])
        if "__context__" not in first:
            raise ValueError("First line must contain '__context__' key.")
        context = first["__context__"]

        # Parse rest as records
        records = [json.loads(line) for line in lines[1:]]

        # Collect all keys that appeared
        all_keys = set()
        for record in records:
            all_keys.update(record.keys())

        tracker = cls(keys=all_keys, context=context, allow_unknown_keys=True)
        tracker._records = records
        return tracker

    def _safe_scalar(self, x):
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        if isinstance(x, pd.DataFrame):
            return f"<DataFrame shape={x.shape}>"
        if isinstance(x, (list, tuple)):
            return str(x)
        if isinstance(x, (np.ndarray)):
            return f"<Array shape={tuple(x.shape)}>"
        if isinstance(x, (torch.Tensor)):
            return f"<Tensor shape={tuple(x.shape)}>"
        return str(type(x))

    def _json_safe(self, x):
        if isinstance(x, pd.DataFrame):
            return x.to_dict(orient="records")
        if isinstance(x, (np.ndarray, torch.Tensor)):
            return x.tolist()
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        if isinstance(x, list):
            return [self._json_safe(item) for item in x]
        if isinstance(x, dict):
            return {k: self._json_safe(v) for k, v in x.items()}
        return str(x)


class LogBook:
    # A simple manager for multiple MetricLogger instances, providing centralized serialization.

    def __init__(self, loggers: list = None):
        # loggers: optional list of MetricLogger instances
        self.loggers = []
        seen_contexts = set()
        if loggers:
            for logger in loggers:
                ctx_key = frozenset(logger._context.items())
                if ctx_key in seen_contexts:
                    raise ValueError(f"Duplicate context detected: {logger._context}")
                seen_contexts.add(ctx_key)
                self.loggers.append(logger)

    def add_logger(self, logger, deepcopy=False) -> None:
        # Add a MetricLogger instance (active) to the LogBook.
        # Ensure its context is unique.
        ctx_key = frozenset(logger._context.items())
        if any(frozenset(lg._context.items()) == ctx_key for lg in self.loggers):
            raise ValueError(f"Logger with context {logger._context} already exists.")
        if deepcopy:
            self.loggers.append(copy.deepcopy(logger))
        else:
            self.loggers.append(logger)

    def record_snapshot(self, logger) -> None:
        # Add a MetricLogger instance (deepcopyed) to the LogBook.
        # so it won't change if the original one is altered.
        self.add_logger(logger, deepcopy=True)

    def find_logger(self, context: dict, exact=True):
        # Return a single logger whose context exactly matches the given context.
        # Raise ValueError if no match or multiple matches are found.
        if exact:
            matched = [logger for logger in self.loggers if logger._context == context]
        else:
            matched = [
                lg
                for lg in self.loggers
                if all(lg._context.get(k) == v for k, v in context.items())
            ]
        if not matched:
            raise ValueError(f"No logger found with context: {context}")
        if len(matched) > 1 and exact:
            raise ValueError(f"Multiple loggers found with context: {context}")
        return matched

    def summary(self):
        rows = []
        for logger in self.loggers:
            ctx = logger._context
            n = len(logger._records)
            row = {**ctx, "n_records": n}
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def __repr__(self):
        df = self.summary()
        if df.empty:
            return "<LogBook (empty)>"
        return f"<LogBook with {len(self.loggers)} loggers>\n" + df.to_string(
            index=False
        )

    def to_dataframe(self) -> pd.DataFrame:
        # Concatenate all loggers' DataFrames,
        # each logger's context as columns included in last few columns
        dfs = []
        for logger in self.loggers:
            dfs.append(logger.to_dataframe())
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def to_json(self, filepath: Optional[str] = None):
        """
        Serialize the entire book to a single JSON, output file at `filepath`.
        Format:
        [
          {"context": {...}, "records": [...]},
          ...
        ]
        """
        output = []
        for logger in self.loggers:
            entry = logger.to_json()
            output.append(entry)
        if filepath:
            with open(filepath, "w") as f:
                json.dump(output, f, indent=2)
        else:
            return output

    def to_pickle(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.loggers, f)

    @classmethod
    def from_pickle(cls, filepath: str):
        with open(filepath, "rb") as f:
            loggers = pickle.load(f)
        return cls(loggers=loggers)

    # def to_pickle(self, filepath: str) -> None:
    #     # Serialize the entire suite to a single pickle file at `filepath`.
    #     data = []
    #     for logger in self.loggers:
    #         context = getattr(logger, "_context", {})
    #         records = getattr(logger, "_records", [])
    #         data.append({"context": context, "records": records})
    #     with open(filepath, "wb") as f:
    #         pickle.dump(data, f)

    # @classmethod
    # def from_pickle(cls, filepath: str, logger_cls):
    #     """
    #     Load a LogBook from a pickle file.

    #     Args:
    #         filepath: path to the pickle file
    #         logger_cls: the MetricLogger class to reconstruct each logger

    #     Returns:
    #         LogBook instance
    #     """
    #     with open(filepath, "rb") as f:
    #         data = pickle.load(f)

    #     loggers = []
    #     for entry in data:
    #         logger = logger_cls(context=entry["context"])
    #         logger._records = entry["records"]
    #         loggers.append(logger)

    #     return cls(loggers=loggers)
