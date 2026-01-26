from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from sarvalanche.grids import GridSpec
from sarvalanche.utils.validation import validate_aoi, validate_canonical, validate_dates
class DataLoader(ABC):
    def __init__(self,
                 product: str,
                 start_time: datetime,
                 stop_time: datetime,
                 reference_grid: GridSpec,
                 cache_dir: Path = Path('./data').resolve()):

        self.product = product
        self.cache_dir = cache_dir
        self.start_time = start_time
        self.stop_time = stop_time
        self.reference_grid = reference_grid

    def get_dataset(self):
        validate_aoi(self.aoi)
        validate_dates(self.start_date, self.stop_date)

        da = self.find_data(self.reference_grid, self.start_time, self.stop_time)


    @abstractmethod
    def find_data(self):
        ...



    def validate_inputs(self):
        validate_aoi(self.aoi)
        validate_dates(self.start_date, self.stop_date)
