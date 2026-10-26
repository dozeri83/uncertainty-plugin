# TODO

## Lens distortion

The plugin projects every training camera as a plain pinhole (focal length and principal point only). For datasets with lens distortion this gives the wrong projection, and so the wrong Fisher blocks and covariances. The camera model enters the covariance through the projection Jacobian `A_i = du_i / dtheta_i`: the 2D mean picks up the distortion Jacobian `J_D`, and the 2D covariance becomes `J_D Sigma_2D J_Dᵀ`. The error is small at the image centre and grows towards the edges.

LichtFeld only trains distorted cameras with `--gut` or `--undistort`, and both need data the plugin does not read today:

- **`--gut`:** the splats were fitted under the distorted projection. `project_u` in `uncertainty/kernels.py` must apply the camera model and distortion coefficients (radial, tangential, model type such as OpenCV or fisheye) and map the 2D covariance through `J_D`. The per-pixel part of the kernel (`dC/du`) does not change, and `A_i` follows automatically because it is computed by finite differences of `project_u`.
- **`--undistort`:** use the undistorted pinhole camera that training used, not the raw one. LichtFeld keeps it in the camera's `UndistortParams` (new K and image size, `core/include/core/camera.hpp`). The raw `focal_x/y` and `center_x/y` the plugin reads now are wrong in this mode.

### Steps

1. **Guard (now):** if any training camera reports `has_distortion`, warn in the panel that the results are approximate, or refuse to compute.
2. **LichtFeld Python bindings:** expose read-only `radial_distortion`, `tangential_distortion`, `camera_model_type` and `undistort_params` on `lichtfeld.scene.Camera` (`src/python/lfs/py_cameras.cpp`), plus which mode training used (GUT on or off, undistort on or off).
3. **Plugin:** use those values. Under GUT, apply the distortion in `project_u`. Under undistort, use the undistorted K and image size, and load the undistorted photos for the noise estimate.

Plugin-only alternative to step 2: read the coefficients from COLMAP `cameras.bin` and recompute the undistorted K. Not preferred: the result must match LichtFeld's undistortion exactly and would break silently if that changes.

## Training loss terms

The Fisher blocks come from a Gauss-Newton approximation of an L2 (Gaussian-noise) photometric loss. LichtFeld trains with L1 + D-SSIM plus regularisers and optional appearance models. Steps to match the real objective, easiest first:

1. **L1 noise model (nearly free).** L1 corresponds to Laplace pixel noise, whose Fisher information is `JᵀJ / b²`, the same form as Gaussian noise. Estimate `b` as the mean absolute residual instead of the RMS. Rankings stay the same; absolute values become right for L1.
2. **Per-splat regularisers (easy, exact).** Opacity and scale regularisation depend only on the splat's own parameters, so add their Hessian directly to its 10×10 block. Read which terms are on and their weights from `lf.optimization_params()`. These terms reduce uncertainty.
3. **D-SSIM (medium).** Generalized Gauss-Newton gives `Jᵀ H_SSIM J`, where `H_SSIM` is SSIM's curvature with respect to pixel colours over its 11×11 window. Practical version: a per-pixel diagonal weight from local image statistics, applied in the existing raster kernel.
4. **Exposure correction (medium).** Marginalise the per-image affine colour parameters `g` with a Schur complement: `F_θθ − F_θg F_gg⁻¹ F_gθ`. This needs a 10×k cross block per splat per view and a small k×k block per image, which fits the per-view loop. It needs the trained exposure parameters, which the Python API does not expose yet. This term increases uncertainty, opacity most.
5. **Bilateral grid and PPISP (hard).** The same marginalisation, but with thousands of parameters per image; the grid's smoothness term enters as a prior on `g`. A low-resolution approximation of the grid is feasible. A faithful version needs the grid tensors (not exposed) and the plugin's renderer applying the grid exactly as LichtFeld does.

The plugin also renders without the appearance models, so the noise estimate is too high when they were used in training. Steps 4–5 fix that too.
