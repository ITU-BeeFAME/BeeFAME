# app/service/analysis_service.py
from collections import defaultdict
import re
import time
from typing import List, Optional
import json
import logging
import unicodedata
from service.redis_client import get_redis_client, cache_set
from service.dataset_service import DatasetService
from model.analysis import ClassifierRequest
from model.classifier import ClassifierName
from model.dataset import DatasetName
from model.evaluation import SubgroupPairRequest
from service.utils.dataset_utils import initial_dataset_analysis, filter_subgroup_rows


class AnalysisService:
    ANALYSIS_VERSION = "v3"

    def __init__(self):
        self.redis_client = get_redis_client()

    def _get_cache_key(self, dataset_name: str, classifier_name: str, test_size: float) -> str:
        # The selected subgroup pairs are deliberately not part of the key: every pair is
        # computed on a miss and the response is filtered, so one fit serves any selection.
        return f"{self.ANALYSIS_VERSION}:{self._slugify(dataset_name)}:{self._slugify(classifier_name)}:{test_size:.4f}"

    def _slugify(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '-', text)
        text = text.strip('-')
        return text

    def analyse(
        self,
        dataset_names: List[DatasetName],
        classifier_requests: List[ClassifierRequest],
        test_size: float = 0.2,
        subgroup_pairs: Optional[List[SubgroupPairRequest]] = None,
    ) -> list:
        results = []
        missing_pairs = []
        requested_labels = {p.label for p in subgroup_pairs} if subgroup_pairs else set()

        for dataset in dataset_names:
            for classifier_request in classifier_requests:
                cache_key = self._get_cache_key(dataset.value, classifier_request.name, test_size)
                try:
                    raw = self.redis_client.get(cache_key)
                    if raw:
                        parsed = json.loads(raw)
                        data = parsed["data"] if isinstance(parsed, dict) and "data" in parsed else parsed
                        results.extend(data)
                    else:
                        missing_pairs.append((dataset, classifier_request))
                except Exception as e:
                    logging.warning(f"Cache read error: {str(e)}")
                    missing_pairs.append((dataset, classifier_request))

        if missing_pairs:
            dataset_service = DatasetService()
            # Each dataset has its own subgroup pairs, so compute one dataset at a time.
            for dataset in dict.fromkeys(ds for ds, _ in missing_pairs):
                classifiers_to_calculate = []
                seen = set()
                for ds, cr in missing_pairs:
                    key = (cr.name, json.dumps(cr.params, sort_keys=True))
                    if ds == dataset and key not in seen:
                        seen.add(key)
                        classifiers_to_calculate.append(cr)

                calculated_results = initial_dataset_analysis(
                    [dataset],
                    classifiers_to_calculate,
                    test_size=test_size,
                    subgroup_pairs=dataset_service.subgroup_pairs_for(dataset.value),
                )

                grouped_data = defaultdict(list)
                for entry in calculated_results:
                    key = (entry["Dataset"], entry["Classifier"])
                    grouped_data[key].append(entry)

                for (dataset_name, classifier), entries in grouped_data.items():
                    cache_key = self._get_cache_key(dataset_name, classifier, test_size)
                    try:
                        wrapped = {"_v": self.ANALYSIS_VERSION, "_ts": int(time.time()), "data": entries}
                        cache_set(self.redis_client, cache_key, json.dumps(wrapped))
                    except Exception as e:
                        logging.warning(f"Cache write error: {str(e)}")
                    results.extend(entries)

        return filter_subgroup_rows(results, requested_labels)
