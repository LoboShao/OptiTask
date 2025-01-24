from simulation.cluster_builder import create_cluster_from_config



def test_simple_cluster_creation():
    # Create cluster from config
    cluster = create_cluster_from_config('testing_configs/simple_config1.yaml')
def test_large_cluster_creation():
    # Create cluster from config
    cluster = create_cluster_from_config('testing_configs/large_config1.yaml')


# Usage example
if __name__ == "__main__":
    test_simple_cluster_creation()