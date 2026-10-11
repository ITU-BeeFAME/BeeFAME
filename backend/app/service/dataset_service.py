# app/service/dataset_service.py
import logging
from repository.dataset_repository import DatasetRepository
from model.dataset import DatasetAnalysis, DatasetInfo, SubgroupPairsResponse
from service.redis_client import get_redis_client, cache_set
from service.utils.dataset_utils import initial_dataset_analysis
from typing import List


class DatasetService:
    SUBGROUP_PAIRS_VERSION = "v2"

    def __init__(self):
        self.redis_client = get_redis_client()

    def get_datasets(self) -> List[DatasetInfo]:
        return DatasetRepository.get_datasets()

    def get_initial_dataset_analysis(self, dataset_id) -> List[DatasetAnalysis]:
        return initial_dataset_analysis(dataset_id)

    def get_subgroup_pairs(self, slug: str) -> SubgroupPairsResponse:
        # Listing the pairs downloads the whole dataset from UCI, so cache the result.
        cache_key = f"{self.SUBGROUP_PAIRS_VERSION}:subgroup-pairs:{slug}"
        try:
            raw = self.redis_client.get(cache_key)
            if raw:
                return SubgroupPairsResponse.model_validate_json(raw)
        except Exception as e:
            logging.warning(f"Cache read error: {str(e)}")

        result = DatasetRepository.get_subgroup_pairs(slug)
        try:
            cache_set(self.redis_client, cache_key, result.model_dump_json())
        except Exception as e:
            logging.warning(f"Cache write error: {str(e)}")
        return result

    def subgroup_pairs_for(self, slug: str) -> List[dict]:
        """Every subgroup pair the dataset offers, in the dict form the analysis code expects."""
        pairs = self.get_subgroup_pairs(slug).pairs
        return [{"col1": p.col1, "col2": p.col2, "col3": p.col3, "label": p.label} for p in pairs]
