from typing import List, Optional, Tuple, Union

import numpy as np
from .transform3d import Transform3d, Translate, Rotate

_R = np.eye(3)  # (3, 3)
_T = np.zeros((1, 3))  # (1, 3)

class CamerasBase:
    def __init__(
        self,        
        **kwargs,
    ) -> None:
        if kwargs is not None:
            for k, v in kwargs.items():                
                setattr(self, k, v)


    def get_projection_transform(self, **kwargs):
        """
        Calculate the projective transformation matrix.
        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.
        Return:
            a `Transform3d` object which represents a batch of projection
            matrices of shape (N, 3, 3)
        """
        raise NotImplementedError()


    def unproject_points(self, xy_depth: np.ndarray, **kwargs):
        """
        Transform input points from camera coodinates (NDC or screen)
        to the world / camera coordinates.
        Each of the input points `xy_depth` of shape (..., 3) is
        a concatenation of the x, y location and its depth.
        For instance, for an input 2D tensor of shape `(num_points, 3)`
        `xy_depth` takes the following form:
            `xy_depth[i] = [x[i], y[i], depth[i]]`,
        for a each point at an index `i`.
        The following example demonstrates the relationship between
        `transform_points` and `unproject_points`:
        .. code-block:: python
            cameras = # camera object derived from CamerasBase
            xyz = # 3D points of shape (num_points, 3)
            # transform xyz to the camera view coordinates
            xyz_cam = cameras.get_world_to_view_transform().transform_points(xyz)
            # extract the depth of each point as the 3rd coord of xyz_cam
            depth = xyz_cam[:, 2:]
            # project the points xyz to the camera
            xy = cameras.transform_points(xyz)[:, :2]
            # append depth to xy
            xy_depth = np.hstack((xy, depth), dim=2)
            # unproject to the world coordinates
            xyz_unproj_world = cameras.unproject_points(xy_depth, world_coordinates=True)
            print(np.allclose(xyz, xyz_unproj_world)) # True
            # unproject to the camera coordinates
            xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)
            print(np.allclose(xyz_cam, xyz_unproj)) # True
        Args:
            xy_depth: torch tensor of shape (..., 3).
            world_coordinates: If `True`, unprojects the points back to world
                coordinates using the camera extrinsics `R` and `T`.
                `False` ignores `R` and `T` and unprojects to
                the camera view coordinates.
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.
        Returns
            new_points: unprojected points with the same shape as `xy_depth`.
        """
        raise NotImplementedError()


    def get_camera_center(self, **kwargs) -> np.ndarray:
        """
        Return the 3D location of the camera optical center
        in the world coordinates.
        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.
        Setting R or T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.
        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        C = P[:, 3, :3]
        return C

    
    def get_world_to_view_transform(self, **kwargs):
        """
        Return the world-to-view transform.
        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.
        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.
        Returns:
            A Transform3d object which represents transform
            of shape (3, 3)
        """
        R = kwargs.get("R", self.R)
        T = kwargs.get("T", self.T)
        self.R = R
        self.T = T
        world_to_view_transform = get_world_to_view_transform(R=R, T=T)
        return world_to_view_transform


    def get_full_projection_transform(self, **kwargs) -> Transform3d:
        """
        Return the full world-to-camera transform composing the
        world-to-view and view-to-camera transforms.
        If camera is defined in NDC space, the projected points are in NDC space.
        If camera is defined in screen space, the projected points are in screen space.
        Args:
            **kwargs: parameters for the projection transforms can be passed in
                as keyword arguments to override the default values
                set in __init__.        
        Returns:
            a Transform3d object which represents a transform
            of shape (3, 3)
        """
        self.R: np.ndarray = kwargs.get("R", self.R)
        self.T: np.ndarray = kwargs.get("T", self.T)
        world_to_view_transform = self.get_world_to_view_transform(R=self.R, T=self.T)
        view_to_proj_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_proj_transform)


    def transform_points(
        self, points, **kwargs
    ) -> np.ndarray:
        """
        Transform input points from world to camera space.
        If camera is defined in NDC space, the projected points are in NDC space.
        If camera is defined in screen space, the projected points are in screen space.        
        Args:
            points: np tensor of shape (..., 3).
        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_proj_transform = self.get_full_projection_transform(**kwargs)
        return world_to_proj_transform.transform_points(points)


    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Returns the transform from camera projection space (screen or NDC) to NDC space.
        For cameras that can be specified in screen space, this transform
        allows points to be converted from screen to NDC space.
        The default transform scales the points from [0, W]x[0, H]
        to [-1, 1]x[-u, u] or [-u, u]x[-1, 1] where u > 1 is the aspect ratio of the image.
        This function should be modified per camera definitions if need be,
        e.g. for Perspective/Orthographic cameras we provide a custom implementation.        
        """
        if self.in_ndc():
            # Identity transform
            return Transform3d(dtype=np.float32)
        else:
            # For custom cameras which can be defined in screen space,
            # users might have to implement the screen to NDC transform based
            # on the definition of the camera parameters.
            # See PerspectiveCameras/OrthographicCameras for an example.
            image_size = kwargs.get("image_size", self.get_image_size())          
            return get_screen_to_ndc_transform(
                self, with_xyflip=False, image_size=image_size                
            )


    def transform_points_ndc(self, points, **kwargs) -> np.ndarray:
        """
        Transforms points from world/camera space to NDC space.
        Input points follow the coordinate system conventions: +X left, +Y up.
        Output points are in NDC space: +X left, +Y up, origin at image center.
        Args:
            points: torch tensor of shape (..., 3).            
        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_ndc_transform = self.get_full_projection_transform(**kwargs)
        if not self.in_ndc():
            to_ndc_transform = self.get_ndc_camera_transform(**kwargs)
            world_to_ndc_transform = world_to_ndc_transform.compose(to_ndc_transform)
        
        return world_to_ndc_transform.transform_points(points)


    def transform_points_screen(
        self, points, **kwargs
    ) -> np.ndarray:
        """
        Transforms points from world/camera space to screen space.
        Input points follow the coordinate system conventions: +X left, +Y up.
        Output points are in screen space: +X right, +Y down, origin at top left corner.
        Args:
            points: numpy array of shape (..., 3).            
        Returns
            new_points: transformed points with the same shape as the input.
        """
        points_ndc = self.transform_points(points, **kwargs)
        image_size = kwargs.get("image_size", self.get_image_size())
        return get_ndc_to_screen_transform(
            image_size=image_size
        ).transform_points(points_ndc)


    def is_perspective(self):
        raise NotImplementedError()


    def in_ndc(self):
        """
        Specifies whether the camera is defined in NDC space
        or in screen (image) space
        """
        raise NotImplementedError()

    
    def get_image_size(self):
        """
        Returns the image size, if provided, expected in the form of (height, width)
        The image size is used for conversion of projected points to screen coordinates.
        """
        return self.image_size if hasattr(self, "image_size") else None



############################################################
#                   Camera Classes                         #
############################################################



class PerspectiveCameras(CamerasBase):
    """    
    Parameters for this camera are specified in NDC if `in_ndc` is set to True.
    If parameters are specified in screen space, `in_ndc` must be set to False.
    """

    def __init__(
        self,
        focal_length = 1.0,
        principal_point = np.array((0.0, 0.0)),
        R: np.ndarray = _R,
        T: np.ndarray = _T,
        K: Optional[np.ndarray] = None,
        in_ndc: bool = True,
        image_size: Optional[Union[List, Tuple, np.ndarray]] = None,
    ) -> None:
        """
        Args:
            focal_length: Focal length of the camera in world units.
                A tensor of shape (1,) or (2,) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (2,).
            in_ndc: True if camera parameters are specified in NDC.
                If camera parameters are in screen space, it must
                be set to False.
            R: Rotation matrix of shape (3, 3)
            T: Translation matrix of shape (3,)
            K: (optional) A calibration matrix of shape (4, 4)
                If provided, don't need focal_length, principal_point
            image_size: (height, width) of image size.
                A tensor of shape (2,) or a list/tuple. Required for screen cameras.            
        """
        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            _in_ndc=in_ndc,
            **kwargs
        )  
        if image_size is not None:
            if (self.image_size < 1):
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        
    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the projection matrix using the
        multi-view geometry convention.
        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in __init__.
        Returns:
            A `Transform3d` object transforms.
        .. code-block:: python
            fx = focal_length[:, 0]
            fy = focal_length[:, 1]
            px = principal_point[:, 0]
            py = principal_point[:, 1]
            K = [
                    [fx,   0,   px,   0],
                    [0,   fy,   py,   0],
                    [0,    0,    1,   0],
                    [0,    0,    0,   1],
            ]
        """
        K = kwargs.get("K", self.K)
        if K is not None:
            if K.shape != (4, 4):
                msg = "Expected K to have shape of (4, 4)"
                raise ValueError(msg)
        else:
            K = _get_calibration_matrix(
                kwargs.get("focal_length", self.focal_length),
                kwargs.get("principal_point", self.principal_point)                
            )

        transform = Transform3d(matrix=K.T)
        return transform


    def unproject_points(
        self, 
        xy_depth: np.ndarray, 
        world_coordinates: bool = True,
        from_ndc: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Args:
            from_ndc: If `False` (default), assumes xy part of input is in
                NDC space if self.in_ndc(), otherwise in screen space. If
                `True`, assumes xy is in NDC space even if the camera
                is defined in screen space.
        """
        if world_coordinates:
            to_camera_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_camera_transform = self.get_projection_transform(**kwargs)
        
        if from_ndc:
            to_camera_transform = to_camera_transform.compose(
                self.get_ndc_camera_transform()
            )
        
        unprojection_transform = to_camera_transform.inverse()      
        return unprojection_transform.transform_points(xy_depth)
        

    def get_principal_point(self, **kwargs) -> np.ndarray:
        """
        Return the camera's principal point
        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.
        """
        proj_matrix = self.get_projection_transform(**kwargs).get_matrix()
        return proj_matrix[:2, 2]

    
    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Returns the transform from camera projection space (screen or NDC) to NDC space.
        If the camera is defined already in NDC space, the transform is identity.
        For cameras defined in screen space, we adjust the principal point computation
        which is defined in the image space (commonly) and scale the points to NDC space.
        This transform leaves the depth unchanged.
        Important: This transforms assumes conventions for the input points +X left, +Y up.
        """
        if self.in_ndc():
            ndc_transform = Transform3d(dtype=np.float32)
        else:
            # when cameras are defined in screen/image space, the principal point is
            # provided in the (+X right, +Y down), aka image, coordinate system.
            # Since input points are defined in the PyTorch3D system (+X left, +Y up),
            # we need to adjust for the principal point transform.
            pr_point_fix = np.zeros(
                (4, 4), dtype=np.float32
            )
            pr_point_fix[0, 0] = 1.0
            pr_point_fix[1, 1] = 1.0
            pr_point_fix[2, 2] = 1.0
            pr_point_fix[3, 3] = 1.0
            pr_point_fix[:2, 3] = -2.0 * self.get_principal_point(**kwargs)
            pr_point_fix_transform = Transform3d(
                matrix=pr_point_fix.T
            )
            image_size = kwargs.get("image_size", self.get_image_size())
            screen_to_ndc_transform = get_screen_to_ndc_transform(
                self, image_size=image_size
            )
            ndc_transform = pr_point_fix_transform.compose(screen_to_ndc_transform)

        return ndc_transform


    def is_perspective(self):
        return True

    
    def in_ndc(self):
        return self._in_ndc



################################################
# Helper functions for world to view transforms
################################################



def _get_calibration_matrix(
    focal_length,
    principal_point,    
) -> np.ndarray:
    """
    Returns a calibration matrix of a perspective/orthographic camera.
    Args:        
        focal_length: Focal length of the camera.
        principal_point: xy coordinates of the center of
            the principal point of the camera in pixels.        
        The calibration matrix `K` is set up as follows:
        .. code-block:: python
            fx = focal_length[:,0]
            fy = focal_length[:,1]
            px = principal_point[:,0]
            py = principal_point[:,1]
            
            K = [
                    [fx,   0,   px,   0],
                    [0,   fy,   py,   0],
                    [0,    0,    1,   0],
                    [0,    0,    0,   1],
            ]
    Returns:
        A calibration matrix `K` of the camera
        of shape ( 4, 4).
    """
    if not isinstance(focal_length, np.ndarray):
        focal_length = np.array([focal_length])
    
    if focal_length.shape == (1,):
        fx = fy = focal_length
    else:
        fx, fy = focal_length

    px, py = principal_point

    K = np.zeros((4, 4))
    K[0, 0] = fx
    K[1, 1] = fy

    K[0, 2] = px
    K[1, 2] = py
    K[2, 2] = 1.0
    K[3, 3] = 1.0

    return K 


def get_world_to_view_transform(
    R: np.array = _R, T: np.array =_T
) -> Transform3d:
    """
    This function returns a Transform3d representing the transformation
    matrix to go from world space to view space by applying a rotation and
    a translation.

    `X_cam = X_world R + T`

    Args:
        R: (3, 3) matrix representing the rotation.
        T: (3) vector representing the translation.
    Returns:
        a Transform3d object which represents the composed RT transformation.
    """

    if T.shape != (3,):
        msg = "Expect T to have shape (3,); got %r"
        raise ValueError(msg % repr(T.shape))
    if R.shape != (3, 3):
        msg = "Expected R to have shape (3, 3); got %r"
        raise ValueError(msg % repr(R.shape))
    
    T_ = Translate(T)
    R_ = Rotate(R)    
    return R_.compose(T_)



def get_ndc_to_screen_transform(      
    image_size: Optional[Union[List, Tuple, np.ndarray]] = None
) -> Transform3d:
    """
    NDC to screen conversion.
    Conversion from NDC space (+X left, +Y up) to screen/image space
    (+X right, +Y down, origin top left).
    Args:
        cameras        
    Optional kwargs:
        image_size: ((height, width),) specifying the height, width
        of the image. If not provided, it reads it from cameras.
    We represent the NDC to screen conversion as a Transform3d
    with projection matrix
    K = [
            [s,   0,    0,  cx],
            [0,   s,    0,  cy],
            [0,   0,    1,   0],
            [0,   0,    0,   1],
    ]
    """
    # We require the image size, which is necessary for the transform
    if image_size is None:
        msg = "For NDC to screen conversion, image_size=(height, width) needs to be specified"
        raise ValueError(msg)

    K = np.zeros((4,4))
    height, width = image_size

    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    scale = min(image_size) / 2.0

    K[0, 0] = scale
    K[1, 1] = scale
    K[0, 3] = -1.0 * width / 2.0
    K[1, 3] = -1.0 * height / 2.0
    K[2, 2] = 1.0
    K[3, 3] = 1.0
    
    return Transform3d(matrix=K.T)

    
def get_screen_to_ndc_transform(
    image_size: Optional[Union[List, Tuple, np.ndarray]] = None
) -> Transform3d:
    """
    Screen to NDC conversion.
    Conversion from screen/image space (+X right, +Y down, origin top left)
    to NDC space (+X left, +Y up).
    Args:
        cameras        
    Optional kwargs:
        image_size: ((height, width),) specifying the height, width
        of the image. If not provided, it reads it from cameras.
    We represent the screen to NDC conversion as a Transform3d
    with projection matrix
    K = [
            [1/s,    0,    0,  cx/s],
            [  0,  1/s,    0,  cy/s],
            [  0,    0,    1,     0],
            [  0,    0,    0,     1],
    ]
    """
    transform = get_ndc_to_screen_transform(
        image_size=image_size
    ).inverse()
    return transform
