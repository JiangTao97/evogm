from omegaconf import DictConfig

from .base_algorithm import ModelFusionAlgorithm

class AlgorithmFactory:
    _algorithms = {
        "evogm": "fusion_bench.method.evogm.EvoGMAlgorithm",
        "evogm_npu": "fusion_bench.method.evogm_npu.EvoGMAlgorithm",
    }

    @staticmethod
    def create_algorithm(method_config: DictConfig) -> ModelFusionAlgorithm:
        from fusion_bench.utils import import_object

        algorithm_name = method_config.name
        if algorithm_name not in AlgorithmFactory._algorithms:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}, available algorithms: {AlgorithmFactory._algorithms.keys()}."
            )
        algorithm_cls = AlgorithmFactory._algorithms[algorithm_name]
        if isinstance(algorithm_cls, str):
            algorithm_cls = import_object(algorithm_cls)
        return algorithm_cls(method_config)

    @staticmethod
    def register_algorithm(name: str, algorithm_cls):
        AlgorithmFactory._algorithms[name] = algorithm_cls

    @classmethod
    def available_algorithms(cls):
        return list(cls._algorithms.keys())


def load_algorithm_from_config(method_config: DictConfig):
    """
    Loads an algorithm based on the provided configuration.

    The function checks the 'name' attribute of the configuration and returns an instance of the corresponding algorithm.
    If the 'name' attribute is not found or does not match any known algorithm names, a ValueError is raised.

    Args:
        method_config (DictConfig): The configuration for the algorithm. Must contain a 'name' attribute that specifies the type of the algorithm.

    Returns:
        An instance of the specified algorithm.

    Raises:
        ValueError: If 'name' attribute is not found in the configuration or does not match any known algorithm names.
    """
    return AlgorithmFactory.create_algorithm(method_config)
