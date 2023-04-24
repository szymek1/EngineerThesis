"""
Abstract class containing an interface for building utilities for image preparation
"""

from abc import ABC, abstractmethod
from typing import List


class IProcess(ABC):

    @abstractmethod
    def get_images(self) -> List[List[str]]:
        """
        :return: list of all images found in given directories
        """
        pass
