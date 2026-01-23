import hydra
from omegaconf import DictConfig
from src.data.factory import generate_and_save_dataset


def run(cfg: DictConfig):
    """Pure entrypoint for unit tests."""
    return generate_and_save_dataset(cfg)


@hydra.main(version_base=None, config_path="../conf/data", config_name="config")
def main(cfg: DictConfig) -> None:
    ds, art = run(cfg)
    print(f"Saved X{ds.X.shape}, y{ds.y.shape} -> {art.data_path}")


if __name__ == "__main__":
    main()
