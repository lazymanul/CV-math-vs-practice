from typing import Optional
import numpy as np


class Transform3d:
    """
    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors. Suppose that t is a Transform3d;
    then we can do the following:
    """

    def __init__(
        self,        
        matrix: Optional[np.ndarray] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """
        Args:
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.            
            matrix: A tensor of shape (4, 4) 
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `dtype`.
        """
        if matrix is None:
            self._matrix = np.eye(4, dtype=dtype)
        else:
            if matrix.ndim != 2:
                raise ValueError('"matrix" has to be a 3-dimentional tensor.')
            if matrix.shape != (4,4):
                raise ValueError('"matrix has to be of shape (4,4)')

            self._matrix = matrix    
            dtype = matrix.dtype
        
        self.dtype = dtype
        self._transforms = []        


    def compose(self, *other_transforms: "Transform3d") -> "Transform3d":
        """
        Return a new Transform3d representing the composition of self with the
        given other transforms, which will be stored as an internal list.
        Args:
            *other_transforms: Any number of Transform3d objects
        Returns:
            A new Transform3d with the stored transforms
        """
        composed = Transform3d(dtype=self.dtype)
        composed._matrix = self._matrix.copy()
        for other_transform in other_transforms:
            if not isinstance(other_transform, Transform3d):
                msg = "Only possible to compose Transform3d objects; got %s"
                raise ValueError(msg % type(other_transform))
        composed._transforms = self._transforms + list(other_transforms)
        return composed


    def get_matrix(self) -> np.ndarray:
        """
        Returns a 4×4 matrix corresponding to transform.
        If the transform was composed from others, the matrix for the composite
        transform will be returned.

        For example, if self.transforms contains transforms t1, t2, and t3, and
        given a set of points x, the following should be true:
        
        .. code-block:: python
            y1 = t1.compose(t2, t3).transform(x)
            y2 = t3.transform(t2.transform(t1.transform(x)))
            y1.get_matrix() == y2.get_matrix()
        
        Returns:
            A (4, 4) transformation matrix representing
            the stored transforms. 
        """
        composed_matrix = self._matrix.copy()
        if len(self._transforms) > 0:
            for other_transform in self._transforms:
                other_matrix = other_transform.get_matrix()
                composed_matrix = other_matrix @ composed_matrix 
        return composed_matrix

    
    def _get_matrix_inverse(self) -> np.ndarray:
        """
        Return the inverse of self._matrix.
        """
        return np.linalg.inv(self._matrix)


    def inverse(self, invert_composed: bool = False) -> "Transform3d":
        """
        Returns a new Transform3d object that represents an inverse of the
        current transformation.
        Args:
            invert_composed:
                - True: First compose the list of stored transformations
                  and then apply inverse to the result. This is
                  potentially slower for classes of transformations
                  with inverses that can be computed efficiently
                  (e.g. rotations and translations).
                - False: Invert the individual stored transformations
                  independently without composing them.
        Returns:
            A new Transform3d object containing the inverse of the original
            transformation.
        """

        tinv = Transform3d(dtype=self.dtype)
        
        if invert_composed:
            # first compose then invert
            tinv._matrix = np.linalg.inv(self.get_matrix())
        else:
            inv_matrix = self._get_matrix_inverse()
            
            if len(self._transforms) > 0:
                # inversed transforms applied in reverse order
                # the last transform is the inversed current one
                tinv._transforms = [t.inverse() for t in reversed(self._transforms)]
                last = Transform3d(dtype=self.dtype)
                last._matrix = inv_matrix 
                tinv._transforms.append(last)
            else:
                tinv._matrix = inv_matrix
        
        return tinv

    def transform_points(self, points) -> np.ndarray:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.
        Args:
            points: Tensor of shape (P, 3)             
        Returns:
            points_out: points of shape (P, 3) depending
            on the dimensions of the transform
        """

        points_batch = points.copy()
        if points_batch.ndim != 2:
            msg = "Expected points to have dim = 2: got shape %r"
            raise ValueError(msg % repr(points.shape))
        
        P, _2 = points_batch.shape
        ones = np.ones((P, 1), dtype=points.dtype)
        points_batch = np.hstack((points_batch, ones))

        composed_matrix = self.get_matrix()
        points_out = points_batch @ composed_matrix.T
        denominator = points_out[..., 3]
        points_out = points_out[..., :3] / denominator
        
        return points_out.reshape(points.shape)

    

class Translate(Transform3d):
    def __init__(
        self, 
        xyz_translation: np.ndarray,
        dtype: np.dtype = np.float32
    ) -> None:
        """
        Create a new Transform3d representing 3D translations.
        Args:
            x, y, z: scalars
        """
        super().__init__(dtype=dtype)
        
        matrix = np.eye(4, dtype=dtype)        
        matrix[:3, 3] = xyz_translation
        self._matrix = matrix


class Scale(Transform3d):
    def __init__(
        self, 
        xyz_scales: np.ndarray,
        dtype: np.dtype = np.float32
    ) -> None:
        """
        A Transform3d representing a scaling operation, with different scale
        factors along each coordinate axis.
        Args:
            x, y, z: scalar axis factors
        """
        super().__init__(dtype=dtype)

        matrix = np.eye(4, dtype=dtype)
        matrix[0, 0] = xyz_scales[0]
        matrix[1, 1] = xyz_scales[1]
        matrix[2, 2] = xyz_scales[2]
        self._matrix = matrix


class Rotate(Transform3d):
    def __init__(
        self, 
        R: np.array, 
        dtype: np.dtype = np.float32
    ) -> None:
        """
        Create a new Transform3d representing 3D rotation using a rotation
        matrix as the input.
        Args:
            R: a tensor of shape (3, 3)
        """
        super().__init__(dtype=dtype)
        
        if R.shape != (3, 3):
            msg = "R must have shape (3,3); got %s"
            raise ValueError(msg % repr(R.shape))

        matrix = np.eye(4, dtype=dtype)
        matrix[:3, :3] = R
        self._matrix = matrix

