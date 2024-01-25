"""SpatialArray module is a wrapper for easier accesing of data elements"""
import numpy as np


class SpatialArray(np.ndarray):
    """
    A subclass of numpy ndarray to handle spatial coordinates more conveniently.
    Supports 1D and 2D arrays representing spatial points or collections of points.
    """

    def __new__(cls, array):
        """
        Create a new instance of SpatialArray.
        """
        obj = np.asarray(array).view(cls)

        return obj

    def __array_finalize__(self, obj):
        """
        Finalizer called when the object is being created.
        """
        if obj is None:
            return

    def _check_dimension(self, dim):
        """
        Helper method to check if the array has the appropriate dimension.
        """
        if self.ndim == 1:
            if len(self) < dim:
                raise ValueError(f"{dim}D coordinate is not available for this array")
        elif self.ndim != 2 or self.shape[1] < dim:
            raise ValueError(f"Unsupported array shape for {dim}D coordinate")

    @property
    def x(self) -> float | np.ndarray:
        """
        Returns the x-coordinate(s) of the array.
        """
        self._check_dimension(1)
        return self[0] if self.ndim == 1 else np.array(self[:, 0])

    @property
    def y(self) -> float | np.ndarray:
        """
        Returns the y-coordinate(s) of the array.
        """
        self._check_dimension(2)
        return self[1] if self.ndim == 1 else np.array(self[:, 1])

    @property
    def z(self) -> float | np.ndarray:
        """
        Returns the z-coordinate(s) of the array.
        """
        self._check_dimension(3)
        return self[2] if self.ndim == 1 else np.array(self[:, 2])
