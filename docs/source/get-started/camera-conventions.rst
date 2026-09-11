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
     - OpenCV and Kalibr (as COLMAP's FAQ states), Kornia
   * - **COLMAP / "half-pixel"**
     - ``(0.5, 0.5)``
     - ``(W/2, H/2)``
     - COLMAP
   * - ``grid_sample``, ``align_corners=True``
     - normalized: corner pixels map to ±1 exactly
     - —
     - torch ``grid_sample``, kornia ``create_meshgrid(normalized_coordinates=True)`` today
   * - ``grid_sample``, ``align_corners=False``
     - normalized: pixel *areas* span [−1, 1]; centers at ±(1−1/N)
     - —
     - torch default since 1.3

The "Used by" column names only projects whose own documentation states the convention — COLMAP's FAQ,
quoted under the frame table, covers OpenCV, Kalibr and COLMAP itself. Learned-reconstruction code often fits
neither row cleanly, so read the code rather than the README. DUSt3R and VGGT enumerate integer pixel
coordinates (`DUSt3R xy_grid
<https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/dust3r/utils/geometry.py#L15-L19>`__,
`VGGT meshgrid
<https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/utils/geometry.py#L107>`__)
but place the principal point at ``(W/2, H/2)`` (`DUSt3R
<https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/dust3r/cloud_opt/init_im_poses.py#L237>`__,
`VGGT
<https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/utils/pose_enc.py#L118-L119>`__),
half a pixel off the first row. MoGe returns intrinsics normalized to the unit square with the principal point at
``(0.5, 0.5)`` (`MoGe v3
<https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v3.py#L311-L319>`__)
on a grid whose first pixel centre is ``(0.5/W, 0.5/H)`` (`utils3d uv_map
<https://github.com/EasternJournalist/utils3d-moge/blob/62f09d58509485564e24d5d9f6aac9ee9ebc0c37/utils3d_moge/torch/maps.py#L37-L76>`__)
— the second row scaled by the image size, which the third rule below converts.

Three rules follow from the first two rows:

- ``cx_colmap = cx_opencv + 0.5``, and the same for ``cy``. Nothing else in the calibration moves: the focal
  lengths and the distortion coefficients are unaffected.
- Resizing an image by a scale ``s`` rescales the principal point differently in each convention — OpenCV
  ``cx' = s * cx + (s - 1) / 2``, COLMAP ``cx' = s * cx``. The two agree only at ``s = 1``. Both formulas assume
  a resampler that keeps the image's outer boundaries in place, which is what
  ``torch.nn.functional.interpolate(align_corners=False)`` and therefore :func:`kornia.geometry.transform.resize`
  at its default do. An ``align_corners=True`` resize keeps the corner pixel *centres* in place instead; the
  integer-centre rule is then ``cx' = cx * (W_out - 1) / (W_in - 1)`` with no offset, and ``fx`` scales by the same
  factor. For 5 → 3 columns, ``cx = 1`` becomes ``0.4`` under the first geometry and ``0.5`` under the second.
- Intrinsics normalized to the unit square with the first pixel centre at ``(0.5/W, 0.5/H)`` are the half-pixel
  convention scaled by the image size. They convert to Kornia's integer-centre pixels as ``fx_px = W * fx_n``,
  ``cx_px = W * cx_n - 0.5``, and the same for ``fy``, ``cy`` with ``H`` (`utils3d denormalize_intrinsics
  <https://github.com/EasternJournalist/utils3d-moge/blob/62f09d58509485564e24d5d9f6aac9ee9ebc0c37/utils3d_moge/torch/transforms.py#L401-L440>`__
  is that matrix). Multiplying by the size alone leaves the half-pixel offset in place.

Kornia's grids are **integer-centre**: :func:`kornia.geometry.grid.create_meshgrid` with
``normalized_coordinates=False`` enumerates ``0 .. W-1`` along ``x`` and ``0 .. H-1`` along ``y``, so a centred
image has its principal point at ``cx = (W - 1) / 2``, ``cy = (H - 1) / 2``. That is what
:func:`kornia.geometry.camera.perspective.project_points`, :func:`kornia.geometry.depth.depth_to_3d` and the
rest of the functional camera API assume.

.. warning::

   :meth:`kornia.geometry.camera.pinhole.PinholeCamera.scale` (and ``scale_``, and
   :meth:`kornia.sensors.camera.PinholeModel.scale`) rescale the principal point as ``cx' = s * cx`` — the COLMAP
   rule — which disagrees with the integer pixel centres the rest of the library uses. It is documented as it
   is and tracked as a coordinated repair in
   `#4263 <https://github.com/kornia/kornia/issues/4263>`_.

Camera and world frames
-----------------------

Each non-Kornia cell below is taken from, or derived from, the upstream documentation linked in the same row,
and the derivations are spelled out under the table; a cell reads "not stated" when that documentation does not
say.

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
     - half-pixel with a **lower-left** window origin: the lower-left pixel is centred at ``(0.5, 0.5)``, and NDC
       ``±1`` are the viewport edges, so OpenGL NDC is the ``align_corners=False`` row above with ``y`` reversed
     - `gluLookAt <https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml>`_,
       `glFrustum <https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml>`_,
       `gl_FragCoord <https://registry.khronos.org/OpenGL-Refpages/gl4/html/gl_FragCoord.xhtml>`_,
       `glViewport <https://registry.khronos.org/OpenGL-Refpages/gl4/html/glViewport.xhtml>`_; the handedness
       from `ARCore Pose <https://developers.google.com/ar/reference/java/com/google/ar/core/Pose>`_
     - for the pose only: :func:`kornia.geometry.conversions.camtoworld_graphics_to_vision_4x4` /
       :func:`kornia.geometry.conversions.camtoworld_graphics_to_vision_Rt` and the inverse pair
       :func:`kornia.geometry.conversions.camtoworld_vision_to_graphics_4x4` /
       :func:`kornia.geometry.conversions.camtoworld_vision_to_graphics_Rt`; the image-coordinate change is under
       the ``grid_sample`` bridge below
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
  in the OpenGL row for that reason. ``gl_FragCoord`` "assumes a lower-left origin for window coordinates and
  assumes pixel centers are located at half-pixel centers. For example, the (x, y) location (0.5, 0.5) is
  returned for the lower-left-most pixel in a window", and ``glViewport`` maps normalized device coordinates to
  window coordinates as ``x_w = (x_nd + 1) * width / 2 + x`` — the edges ``±1`` land on the viewport edges, not
  on the outer pixel centres.
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
- :class:`kornia.geometry.camera.pinhole.PinholeCamera` instead stores a ``4x4`` ``intrinsics`` whose canonical
  form is the homogeneous embedding ``[[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]``. The whole
  matrix participates: :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.project` multiplies the full
  ``intrinsics @ extrinsics`` and :meth:`~kornia.geometry.camera.pinhole.PinholeCamera.unproject` inverts that
  ``4x4`` product, so a non-zero ``intrinsics[0, 3]`` shifts every projected ``u`` by ``intrinsics[0, 3] / z``,
  ``intrinsics[3, 3]`` rescales every unprojected point, and a ``3x3`` ``K`` zero-padded without
  ``intrinsics[3, 3] = 1`` makes ``unproject`` fail on a singular matrix. Its class docstring documents the
  layout and the deviations.
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

OpenGL's normalized device coordinates are the ``align_corners=False`` reading with the ``y`` axis reversed,
because the window origin is the lower-left corner: in a 5 × 3 image the centre of the top-left pixel is
``(-0.8, +2/3)`` in OpenGL NDC, ``(-0.8, -2/3)`` under ``align_corners=False`` and ``(-1, -1)`` under
``align_corners=True``. The pose converters in the frames table change the camera basis only. A principal point
expressed in OpenGL window coordinates is half-pixel with ``y`` up: flip it with ``cy → H - cy``, then subtract
``0.5`` from both coordinates — the COLMAP rule — to obtain Kornia's integer-centre principal point.

Three camera-adjacent warps sample with ``align_corners=True``:

- :func:`kornia.geometry.calibration.undistort_image` has no ``align_corners`` parameter and remaps with
  ``align_corners=True``.
- :func:`kornia.geometry.depth.warp_frame_depth` has no ``align_corners`` parameter and samples with
  ``align_corners=True``.
- :class:`kornia.geometry.depth.DepthWarper` and :func:`kornia.geometry.depth.depth_warp` do expose it,
  defaulting to ``True``.

The library-wide table of ``align_corners`` defaults is on :doc:`conventions`.
