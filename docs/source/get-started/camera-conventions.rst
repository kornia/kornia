Camera and world conventions across the ecosystem
=================================================

.. meta::
   :description: How Kornia's pixel centres, camera frames, extrinsics and intrinsics relate to OpenCV, COLMAP,
      OpenGL, ARKit, ARCore, PyTorch3D and Direct3D, and which Kornia converter crosses each boundary.

Kornia's camera stack follows the OpenCV pinhole model: pixel coordinates are ``(u, v)`` = ``(x, y)`` =
(column, row) with **integer pixel centres**, ``extrinsics`` is the **world-to-camera** transform, and ``depth``
is the camera-frame ``z``. Several of the ecosystems below differ on at least one of those, and the difference
is silent: the shapes still match, the images still look plausible, and the reconstruction is merely half a
pixel off, or mirrored.

Read :doc:`conventions` first — it holds the library-wide tensor, coordinate and ``align_corners`` rules. This
page covers what changes when a calibration or a pose crosses a library boundary.

Pixel-centre conventions
------------------------

.. list-table::
   :header-rows: 1

   * - Convention
     - Pixel (0, 0) center at
     - Principal point of centered W×H image
     - Used by
   * - **OpenCV / "integer"**
     - ``(0.0, 0.0)``
     - ``((W-1)/2, (H-1)/2)``
     - OpenCV, DUSt3R, VGGT, MoGe, most DL repos
   * - **COLMAP / "half-pixel"**
     - ``(0.5, 0.5)``
     - ``(W/2, H/2)``
     - COLMAP, pycolmap, many SfM tools
   * - ``grid_sample``, ``align_corners=True``
     - normalized: corner pixels map to ±1 exactly
     - —
     - torch ``grid_sample``, kornia ``create_meshgrid(normalized_coordinates=True)`` today
   * - ``grid_sample``, ``align_corners=False``
     - normalized: pixel *areas* span [−1, 1]; centers at ±(1−1/N)
     - —
     - torch default since 1.3

The "Used by" column names the projects each convention is commonly associated with. It is reproduced from
the proposal this table comes from, and — unlike the frame table below — those attributions were not checked
against upstream documentation.

Two rules follow from the first two rows:

- ``cx_colmap = cx_opencv + 0.5``, and the same for ``cy``. Nothing else in the calibration moves: the focal
  lengths and the distortion coefficients are unaffected.
- Resizing an image by a scale ``s`` rescales the principal point differently in each convention — OpenCV
  ``cx' = s * cx + (s - 1) / 2``, COLMAP ``cx' = s * cx``. The two agree only at ``s = 1``.

Kornia's grids are **integer-centre**: :func:`kornia.geometry.grid.create_meshgrid` with
``normalized_coordinates=False`` enumerates ``0 .. W-1`` along ``x`` and ``0 .. H-1`` along ``y``, so a centred
image has its principal point at ``cx = (W - 1) / 2``, ``cy = (H - 1) / 2``. That is what
:func:`kornia.geometry.camera.perspective.project_points`, :func:`kornia.geometry.depth.depth_to_3d` and the
rest of the functional camera API assume. The convention is pinned by:

- ``tests/geometry/camera/test_perspective.py``,
  ``TestProjectPoints::test_convention_integer_pixel_centres_put_the_principal_point_at_w_minus_one_half``
- ``tests/geometry/test_depth.py``,
  ``TestDepthTo3d::test_convention_pixel_origin_is_the_integer_centre``

.. warning::

   :meth:`kornia.geometry.camera.pinhole.PinholeCamera.scale` (and ``scale_``, and
   ``kornia.sensors.camera.PinholeModel.scale``) rescale the principal point as ``cx' = s * cx`` — the COLMAP
   rule — which disagrees with the integer pixel centres the rest of the library uses. It is documented as it
   is and tracked as a coordinated repair in
   `#4263 <https://github.com/kornia/kornia/issues/4263>`_.

Camera and world frames
-----------------------

Each non-Kornia cell below is taken from, or derived from, the upstream documentation linked in the same row,
and the derivations are spelled out under the table; a cell reads "not stated" when that documentation does not
say, and "not verified" when the page could not be read.

.. list-table::
   :header-rows: 1

   * - Convention
     - Camera axes (right / up / forward)
     - Handedness
     - World up
     - Pixel centre
     - Upstream reference
     - Kornia converter
   * - **OpenCV** (Kornia's own frame)
     - +X right, +Y down, +Z forward
     - right-handed
     - not stated
     - integer
     - `OpenCV calib3d <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>`_
     - native — no conversion
   * - **COLMAP**
     - +X right, +Y bottom, +Z front
     - right-handed
     - not stated
     - half-pixel
     - `COLMAP output format <https://colmap.github.io/format.html>`_,
       `camera models <https://colmap.github.io/cameras.html>`_,
       `FAQ <https://colmap.github.io/faq.html>`_
     - :func:`kornia.geometry.conversions.camtoworld_to_worldtocam_Rt` and
       :func:`kornia.geometry.conversions.worldtocam_to_camtoworld_Rt` for the pose;
       :func:`kornia.geometry.conversions.ARKitQTVecs_to_ColmapQTVecs` from ARKit; the principal-point rule
       above has no helper
   * - **OpenGL**
     - +X right, +Y up, −Z forward
     - right-handed
     - set by the ``gluLookAt`` up vector
     - normalized device coordinates; see the ``grid_sample`` rows above
     - `gluLookAt <https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml>`_,
       `glFrustum <https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml>`_; the handedness
       from `ARCore Pose <https://developers.google.com/ar/reference/java/com/google/ar/core/Pose>`_
     - :func:`kornia.geometry.conversions.camtoworld_graphics_to_vision_4x4` /
       :func:`kornia.geometry.conversions.camtoworld_graphics_to_vision_Rt` and the inverse pair
       :func:`kornia.geometry.conversions.camtoworld_vision_to_graphics_4x4` /
       :func:`kornia.geometry.conversions.camtoworld_vision_to_graphics_Rt`
   * - **ARKit**
     - +X right, +Y up, +Z away from the device on the screen side — the camera looks along −Z
     - right-handed
     - oriented by the session configuration
     - not stated
     - `ARCamera.transform <https://developer.apple.com/documentation/arkit/arcamera/transform>`_
     - :func:`kornia.geometry.conversions.ARKitQTVecs_to_ColmapQTVecs`
   * - **ARCore**
     - +X right, +Y up, −Z in the direction the camera is looking
     - right-handed
     - not stated
     - not stated
     - `ARCore Camera <https://developers.google.com/ar/reference/java/com/google/ar/core/Camera>`_,
       `Pose <https://developers.google.com/ar/reference/java/com/google/ar/core/Pose>`_
     - none — converter candidate
   * - **PyTorch3D**
     - +X left, +Y up, +Z from us to the scene
     - right-handed
     - not stated
     - corner-based: ``(0, 0)`` is the top-left corner of the top-left pixel
     - `PyTorch3D cameras <https://pytorch3d.org/docs/cameras>`_
     - none — converter candidate
   * - **Direct3D**
     - +X right, +Y up; +Z follows from the left-hand rule
     - left-handed
     - not stated
     - not stated
     - `Coordinate Systems (Direct3D 9)
       <https://learn.microsoft.com/en-us/windows/win32/direct3d9/coordinate-systems>`_
     - none — converter candidate
   * - **Blender**
     - not verified
     - not verified
     - not verified
     - not verified
     - not verified
     - none — converter candidate

What each row rests on:

- OpenCV states the projection as ``u = fx * Xc/Zc + cx``, ``v = fy * Yc/Zc + cy`` with the coordinates
  ``(u, v)`` "measured in pixels from the top-left corner of the image", and its cheirality check "means that
  the triangulated 3D points should have positive depth". ``+Xc`` therefore grows with the column index,
  ``+Yc`` with the row index, and a point in front of the camera has ``Zc > 0``; the handedness follows from
  those three directions. OpenCV's own prose never names the axis directions, so that cell is **derived** from
  these three quotes rather than quoted. The pixel-centre cell is COLMAP's statement about OpenCV, quoted
  below.
- COLMAP: "The local camera coordinate system of an image is defined in a way that the X axis points to the
  right, the Y axis to the bottom, and the Z axis to the front as seen from the image", it "uses a corner-based
  pixel convention, in which the center of the top-left pixel is at ``(0.5, 0.5)``", and — about the other
  convention — "OpenCV and Kalibr place integer coordinates at pixel *centers*, so their centered principal
  point is ``((width - 1) / 2, (height - 1) / 2)``".
- OpenGL: ``gluLookAt`` "maps the reference point to the negative z axis and the eye point to the origin", and
  the up vector "is mapped to the positive y axis so that it points upward in the viewport"; ``glFrustum``
  places the near plane at ``-nearVal`` with ``nearVal`` positive, "assuming that the eye is located at
  (0, 0, 0)". No fetched Khronos page states the handedness, so that cell is ARCore's characterisation of
  OpenGL — "Coordinate system is right-handed, like OpenGL conventions" — and its ``Pose`` reference is linked
  in the OpenGL row for that reason.
- ARKit: "the x-axis points to the right when the device is in landscapeLeft orientation […] The y-axis points
  upward (with respect to landscapeLeft orientation), and the z-axis points away from the device on the screen
  side" — the screen side faces the user, so the rear camera's viewing direction is ``-Z``; world space
  "follows a right-handed convention, but is oriented based on the session configuration".
- ARCore: the camera pose has "+X pointing right, +Y pointing up, and -Z pointing in the direction the camera
  is looking", and its coordinate system "is right-handed, like OpenGL conventions".
- PyTorch3D: "+X:left", "+Y: up" and "+Z: from us to scene (right-handed)"; in screen coordinates "(0,0) is the
  top left corner of the top left pixel".
- Direct3D: "In both coordinate systems, the positive x-axis points to the right, and the positive y-axis
  points up", and "Direct3D uses a left-handed coordinate system".
- Blender: ``docs.blender.org`` refused the request when this page was written, so no cell is filled from
  memory.

Extrinsics semantics
--------------------

``extrinsics`` in Kornia is the **world-to-camera** ``[R | t]``, the OpenCV and COLMAP semantics:
:meth:`kornia.geometry.camera.pinhole.PinholeCamera.project` takes world points and computes ``K (R X + t)``.
COLMAP's ``images.txt`` stores the same direction — "the projection from world to the camera coordinate system
of an image" — so its ``QW QX QY QZ TX TY TZ`` needs no inversion, only the quaternion-to-matrix step.

The inverse is the **camera-to-world** pose, whose translation column is the camera centre in world
coordinates. Convert in either direction with :func:`kornia.geometry.conversions.worldtocam_to_camtoworld_Rt` and
:func:`kornia.geometry.conversions.camtoworld_to_worldtocam_Rt` — their ``Convention:`` blocks document which
``t`` means what on each side, and that both compute the rigid inverse ``(R^T, -R^T t)`` by transposition.

The graphics/vision converters take a **camera-to-world** pose, not a world-to-camera one; feeding the wrong
direction in flips the wrong side of the product and is a silent error rather than an exception. Their
docstrings carry the details.

Intrinsics layout
-----------------

- The functional API (:func:`kornia.geometry.camera.perspective.project_points`,
  :func:`kornia.geometry.depth.depth_to_3d`, :func:`kornia.geometry.calibration.undistort_points`, …) takes a
  row-major ``3x3`` ``K`` with ``fx = K[0, 0]``, ``fy = K[1, 1]``, ``cx = K[0, 2]``, ``cy = K[1, 2]``, and
  :class:`kornia.sensors.camera.CameraModelBase` exposes that same layout through its ``K()`` method. The skew
  entry ``K[0, 1]`` is ignored by
  :func:`kornia.geometry.conversions.normalize_points_with_intrinsics` and its inverse.
- :class:`kornia.geometry.camera.pinhole.PinholeCamera` instead stores a ``4x4`` ``intrinsics``, of which the
  top-left ``3x3`` block is read. Its class docstring documents the layout and the deviations.
- **Depth means two different things.** It is the camera-frame ``z`` by default, and the Euclidean ray length
  when :func:`kornia.geometry.camera.perspective.unproject_points` is called with ``normalize=True`` (the
  ``normalize_points`` flags of :func:`kornia.geometry.depth.depth_to_3d` and
  :func:`kornia.geometry.depth.depth_to_3d_v2` read it the same way). Monocular-depth networks disagree on
  which one they predict; the choice moves every unprojected point that is not on the principal ray.

``grid_sample`` bridge
----------------------

Pixel coordinates reach :func:`torch.nn.functional.grid_sample` through normalized ``[-1, 1]`` coordinates, and
the last two rows of the pixel-centre table are the two ways to build them. Kornia's normalized coordinates are
the ``align_corners=True`` reading — :func:`kornia.geometry.grid.create_meshgrid` returns that grid by default — so
a camera calibration expressed with integer pixel centres maps onto it without an offset, while the
``align_corners=False`` reading shifts by half a pixel at the image borders.

Three camera-adjacent warps sample with ``align_corners=True``:

- :func:`kornia.geometry.calibration.undistort_image` has no ``align_corners`` parameter and remaps with
  ``align_corners=True``.
- :func:`kornia.geometry.depth.warp_frame_depth` has no ``align_corners`` parameter and samples with
  ``align_corners=True``.
- ``DepthWarper`` and ``depth_warp`` do expose it, defaulting to ``True``.

The library-wide table of ``align_corners`` defaults is on :doc:`conventions`.
