"""CUDA kernels for the per-Gaussian Fisher pass, compiled at runtime with NVRTC (CuPy).

The pass renders each training view with a small tile rasterizer that follows the
LichtFeld FastGS conventions (EWA projection, 0.3 px dilation, pixel centres at +0.5,
alpha in [1/255, 0.999], transmittance cut at 1e-4) and, per pixel, accumulates the
Gauss-Newton (Fisher) block of every Gaussian:

    F_i += sum_pixels J_p,i^T J_p,i,   J_p,i = dC_p / dtheta_i  (3 x 10)

The pixel colour depends on Gaussian i only through its alpha, so J = (dC/dalpha) g^T
with g = dalpha/du, u = (mean2d.x, mean2d.y, conic.a, conic.b, conic.c, opacity logit).
Per pixel the contribution to the 6x6 block in u-space is |dC/dalpha|^2 g g^T. After a
view is rasterized, each Gaussian maps that block to its parameters with the projection
Jacobian A = du/dtheta (6 x 10, central differences in double precision):

    F_i += A^T M_i A

theta = (mean xyz, log-scale xyz, local rotation xyz, opacity logit).
"""

KERNEL_SOURCE = r"""
#define TILE 16
#define BLOCK (TILE * TILE)
#define NP 10          // parameters per Gaussian
#define NU 6           // screen-space parameters per Gaussian
#define NF 55          // packed upper triangle of NP x NP
#define NM 21          // packed upper triangle of NU x NU
#define MIN_ALPHA (1.0f / 255.0f)
#define MAX_ALPHA 0.999f
#define T_EPS 1e-4f
#define DILATION 0.3f

struct Cam {
    float R[9];     // world-to-camera rotation, row-major
    float t[3];     // world-to-camera translation
    float campos[3];
    float fx, fy, cx, cy;
    float clip_l, clip_r, clip_t, clip_b;
    float near_plane;
    int width, height, tiles_x, tiles_y;
};

__device__ __constant__ float SH_C0 = 0.28209479177387814f;
__device__ __constant__ float SH_C1 = 0.4886025119029199f;
__device__ __constant__ float SH_C2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                                         -1.0925484305920792f, 0.5462742152960396f};
__device__ __constant__ float SH_C3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                                         0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                                         -0.5900435899266435f};

__device__ inline int triu(int r, int c, int n) {
    // packed index of (r, c) with r <= c in an n x n upper triangle
    return r * n - (r * (r - 1)) / 2 + (c - r);
}

// ---------------------------------------------------------------------------------
// Projection of one Gaussian to screen-space parameters u = (mx, my, a, b, c).
// Templated on precision so the Jacobian can use double.
// ---------------------------------------------------------------------------------
template <typename T>
__device__ inline void quat_to_rot(const T q[4], T R[9]) {
    T w = q[0], x = q[1], y = q[2], z = q[3];
    T n = sqrt(w * w + x * x + y * y + z * z);
    n = n > T(1e-12) ? n : T(1e-12);
    w /= n; x /= n; y /= n; z /= n;
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);     R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);     R[8] = 1 - 2 * (x * x + y * y);
}

template <typename T>
__device__ inline void rodrigues(const T w[3], T R[9]) {
    T th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    T th = sqrt(th2);
    T a, b;
    if (th < T(1e-8)) { a = 1 - th2 / 6; b = T(0.5) - th2 / 24; }
    else { a = sin(th) / th; b = (1 - cos(th)) / th2; }
    T K[9] = {0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0};
    T K2[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            T s = 0;
            for (int k = 0; k < 3; ++k) s += K[i * 3 + k] * K[k * 3 + j];
            K2[i * 3 + j] = s;
        }
    for (int i = 0; i < 9; ++i) R[i] = (i % 4 == 0 ? T(1) : T(0)) + a * K[i] + b * K2[i];
}

// theta9 = (mean xyz, log-scale xyz, local rotation xyz); base quaternion q0
// returns false if behind the near plane or degenerate
template <typename T>
__device__ inline bool project_u(const Cam& cam, const T theta[9], const T q0[4], T u[5], T* depth_out) {
    const T* m = theta;
    T tc[3];
    for (int r = 0; r < 3; ++r)
        tc[r] = T(cam.R[r * 3 + 0]) * m[0] + T(cam.R[r * 3 + 1]) * m[1] + T(cam.R[r * 3 + 2]) * m[2] + T(cam.t[r]);
    T z = tc[2];
    if (z < T(cam.near_plane)) return false;
    T iz = T(1) / z;
    T x = tc[0] * iz, y = tc[1] * iz;

    T Rq[9], Rd[9], Rf[9];
    quat_to_rot(q0, Rq);
    rodrigues(theta + 6, Rd);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            T s = 0;
            for (int k = 0; k < 3; ++k) s += Rq[i * 3 + k] * Rd[k * 3 + j];
            Rf[i * 3 + j] = s;
        }
    T var[3];
    for (int k = 0; k < 3; ++k) { T sc = exp(theta[3 + k]); var[k] = sc * sc; }
    // Sigma = Rf diag(var) Rf^T
    T S[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S[i * 3 + j] = Rf[i * 3 + 0] * var[0] * Rf[j * 3 + 0] + Rf[i * 3 + 1] * var[1] * Rf[j * 3 + 1] +
                           Rf[i * 3 + 2] * var[2] * Rf[j * 3 + 2];
    T tx = x < T(cam.clip_l) ? T(cam.clip_l) : (x > T(cam.clip_r) ? T(cam.clip_r) : x);
    T ty = y < T(cam.clip_t) ? T(cam.clip_t) : (y > T(cam.clip_b) ? T(cam.clip_b) : y);
    T j11 = T(cam.fx) * iz, j13 = -j11 * tx, j22 = T(cam.fy) * iz, j23 = -j22 * ty;
    // JW rows (2 x 3)
    T a1[3], a2[3];
    for (int k = 0; k < 3; ++k) {
        a1[k] = j11 * T(cam.R[0 * 3 + k]) + j13 * T(cam.R[2 * 3 + k]);
        a2[k] = j22 * T(cam.R[1 * 3 + k]) + j23 * T(cam.R[2 * 3 + k]);
    }
    T c11 = 0, c12 = 0, c22 = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            c11 += a1[i] * S[i * 3 + j] * a1[j];
            c12 += a1[i] * S[i * 3 + j] * a2[j];
            c22 += a2[i] * S[i * 3 + j] * a2[j];
        }
    c11 += T(DILATION);
    c22 += T(DILATION);
    T det = c11 * c22 - c12 * c12;
    if (det < T(1e-6)) return false;
    T idet = T(1) / det;
    u[0] = x * T(cam.fx) + T(cam.cx);
    u[1] = y * T(cam.fy) + T(cam.cy);
    u[2] = c22 * idet;
    u[3] = -c12 * idet;
    u[4] = c11 * idet;
    if (depth_out) *depth_out = z;
    return true;
}

// ---------------------------------------------------------------------------------
// 1. per-Gaussian preprocess: screen-space parameters, colour and tile footprint
// ---------------------------------------------------------------------------------
extern "C" __global__ void preprocess(
    const int N, const Cam* __restrict__ camp,
    const float* __restrict__ means, const float* __restrict__ scales, const float* __restrict__ quats,
    const float* __restrict__ opac_raw, const float* __restrict__ sh0, const float* __restrict__ shN,
    const int n_rest, const int degree,
    float2* __restrict__ xy, float* __restrict__ conic, float* __restrict__ opac, float* __restrict__ color,
    float* __restrict__ depth, int4* __restrict__ rect, int* __restrict__ tiles_touched)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    tiles_touched[i] = 0;
    const Cam cam = *camp;
    float theta[9] = {means[i * 3], means[i * 3 + 1], means[i * 3 + 2],
                      fminf(scales[i * 3], 20.f), fminf(scales[i * 3 + 1], 20.f), fminf(scales[i * 3 + 2], 20.f),
                      0.f, 0.f, 0.f};
    float q[4] = {quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]};
    float u[5], z;
    if (!project_u<float>(cam, theta, q, u, &z)) return;
    float o = 1.f / (1.f + expf(-opac_raw[i]));
    if (o < MIN_ALPHA) return;
    // covariance back from the conic for the footprint
    float det_c = u[2] * u[4] - u[3] * u[3];
    if (det_c <= 0.f) return;
    float c11 = u[4] / det_c, c22 = u[2] / det_c, c12 = -u[3] / det_c;
    float pt = sqrtf(2.f * logf(o * 255.f));
    float ex = pt * sqrtf(c11), ey = pt * sqrtf(c22);
    int x0 = max(0, (int)floorf((u[0] - ex) / TILE));
    int x1 = min(cam.tiles_x, (int)floorf((u[0] + ex) / TILE) + 1);
    int y0 = max(0, (int)floorf((u[1] - ey) / TILE));
    int y1 = min(cam.tiles_y, (int)floorf((u[1] + ey) / TILE) + 1);
    if (x1 <= x0 || y1 <= y0) return;
    (void)c12;

    // view-dependent colour (3DGS convention, clamp at 0)
    float dx = means[i * 3] - cam.campos[0], dy = means[i * 3 + 1] - cam.campos[1], dz = means[i * 3 + 2] - cam.campos[2];
    float n = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-20f);
    float X = dx * n, Y = dy * n, Z = dz * n;
    float basis[16];
    basis[0] = SH_C0;
    int nb = 1;
    if (degree >= 1) { basis[1] = -SH_C1 * Y; basis[2] = SH_C1 * Z; basis[3] = -SH_C1 * X; nb = 4; }
    if (degree >= 2) {
        float xx = X * X, yy = Y * Y, zz = Z * Z, xy_ = X * Y, yz = Y * Z, xz = X * Z;
        basis[4] = SH_C2[0] * xy_; basis[5] = SH_C2[1] * yz; basis[6] = SH_C2[2] * (2.f * zz - xx - yy);
        basis[7] = SH_C2[3] * xz; basis[8] = SH_C2[4] * (xx - yy); nb = 9;
        if (degree >= 3) {
            basis[9] = SH_C3[0] * Y * (3.f * xx - yy); basis[10] = SH_C3[1] * xy_ * Z;
            basis[11] = SH_C3[2] * Y * (4.f * zz - xx - yy); basis[12] = SH_C3[3] * Z * (2.f * zz - 3.f * xx - 3.f * yy);
            basis[13] = SH_C3[4] * X * (4.f * zz - xx - yy); basis[14] = SH_C3[5] * Z * (xx - yy);
            basis[15] = SH_C3[6] * X * (xx - 3.f * yy); nb = 16;
        }
    }
    if (nb - 1 > n_rest) nb = n_rest + 1;
    for (int ch = 0; ch < 3; ++ch) {
        float c = basis[0] * sh0[i * 3 + ch];
        for (int k = 1; k < nb; ++k) c += basis[k] * shN[(i * n_rest + (k - 1)) * 3 + ch];
        color[i * 3 + ch] = fmaxf(c + 0.5f, 0.f);
    }
    xy[i] = make_float2(u[0], u[1]);
    conic[i * 3] = u[2]; conic[i * 3 + 1] = u[3]; conic[i * 3 + 2] = u[4];
    opac[i] = o;
    depth[i] = z;
    rect[i] = make_int4(x0, y0, x1, y1);
    tiles_touched[i] = (x1 - x0) * (y1 - y0);
}

// ---------------------------------------------------------------------------------
// 2. duplicate Gaussians per touched tile with a (tile | depth) sort key
// ---------------------------------------------------------------------------------
extern "C" __global__ void duplicate_keys(
    const int N, const int tiles_x, const int* __restrict__ offsets, const int* __restrict__ tiles_touched,
    const int4* __restrict__ rect, const float* __restrict__ depth,
    unsigned long long* __restrict__ keys, int* __restrict__ vals)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || tiles_touched[i] == 0) return;
    int off = offsets[i] - tiles_touched[i];  // offsets is an inclusive scan
    int4 r = rect[i];
    unsigned int dbits = __float_as_uint(depth[i]);
    for (int y = r.y; y < r.w; ++y)
        for (int x = r.x; x < r.z; ++x) {
            unsigned long long key = ((unsigned long long)(y * tiles_x + x) << 32) | dbits;
            keys[off] = key;
            vals[off] = i;
            ++off;
        }
}

extern "C" __global__ void tile_ranges(const int L, const unsigned long long* __restrict__ keys, int2* __restrict__ ranges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;
    int tile = (int)(keys[idx] >> 32);
    if (idx == 0) ranges[tile].x = 0;
    else {
        int prev = (int)(keys[idx - 1] >> 32);
        if (tile != prev) { ranges[prev].y = idx; ranges[tile].x = idx; }
    }
    if (idx == L - 1) ranges[tile].y = L;
}

// ---------------------------------------------------------------------------------
// 3. rasterize one view; optionally accumulate per-Gaussian 6x6 screen-space Fisher
// ---------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(BLOCK) raster_fisher(
    const Cam* __restrict__ camp, const int2* __restrict__ ranges, const int* __restrict__ point_list,
    const float2* __restrict__ xy, const float* __restrict__ conic, const float* __restrict__ opac,
    const float* __restrict__ color, const float bg_r, const float bg_g, const float bg_b,
    float* __restrict__ out_img, float* __restrict__ M, float* __restrict__ Wsum, const int compute_fisher)
{
    const Cam& cam = *camp;
    const int tile = blockIdx.x;
    const int tx = tile % cam.tiles_x, ty = tile / cam.tiles_x;
    const int lx = threadIdx.x % TILE, ly = threadIdx.x / TILE;
    const int px = tx * TILE + lx, py = ty * TILE + ly;
    const bool inside = px < cam.width && py < cam.height;
    const float fx_ = px + 0.5f, fy_ = py + 0.5f;
    const int2 range = ranges[tile];
    const int lane = threadIdx.x & 31;

    __shared__ int s_id[BLOCK];
    __shared__ float2 s_xy[BLOCK];
    __shared__ float4 s_co[BLOCK];   // conic a, b, c, opacity
    __shared__ float3 s_col[BLOCK];
    __shared__ int s_maxlast;

    // ---- forward ----
    float T = 1.f, C0 = 0.f, C1 = 0.f, C2 = 0.f;
    int last = range.x;
    bool done = !inside;
    const int rounds = (range.y - range.x + BLOCK - 1) / BLOCK;
    int todo = range.y - range.x;
    for (int r = 0; r < rounds; ++r, todo -= BLOCK) {
        if (__syncthreads_count(done) == BLOCK) break;
        int p = range.x + r * BLOCK + threadIdx.x;
        if (p < range.y) {
            int id = point_list[p];
            s_id[threadIdx.x] = id;
            s_xy[threadIdx.x] = xy[id];
            s_co[threadIdx.x] = make_float4(conic[id * 3], conic[id * 3 + 1], conic[id * 3 + 2], opac[id]);
            s_col[threadIdx.x] = make_float3(color[id * 3], color[id * 3 + 1], color[id * 3 + 2]);
        }
        __syncthreads();
        int cnt = min(BLOCK, todo);
        for (int j = 0; !done && j < cnt; ++j) {
            float dx = s_xy[j].x - fx_, dy = s_xy[j].y - fy_;
            float4 co = s_co[j];
            float power = -0.5f * (co.x * dx * dx + co.z * dy * dy) - co.y * dx * dy;
            if (power > 0.f) continue;
            float alpha = fminf(MAX_ALPHA, co.w * __expf(power));
            if (alpha < MIN_ALPHA) continue;
            float Tn = T * (1.f - alpha);
            if (Tn < T_EPS) { done = true; continue; }
            float w = alpha * T;
            C0 += s_col[j].x * w; C1 += s_col[j].y * w; C2 += s_col[j].z * w;
            T = Tn;
            last = range.x + r * BLOCK + j + 1;
        }
    }
    if (inside) {
        int pix = py * cam.width + px;
        out_img[pix * 3 + 0] = C0 + T * bg_r;
        out_img[pix * 3 + 1] = C1 + T * bg_g;
        out_img[pix * 3 + 2] = C2 + T * bg_b;
    }
    if (!compute_fisher) return;

    // ---- backward-order Fisher accumulation, all threads in lockstep ----
    if (threadIdx.x == 0) s_maxlast = range.x;
    __syncthreads();
    if (inside) atomicMax(&s_maxlast, last);
    __syncthreads();
    const int maxlast = s_maxlast;
    const int my_last = inside ? last : range.x;
    float Tcur = T;                                  // transmittance after the current Gaussian
    float S0 = T * bg_r, S1 = T * bg_g, S2 = T * bg_b;  // colour behind the current Gaussian

    for (int base = maxlast; base > range.x; base -= BLOCK) {
        const int cnt = min(BLOCK, base - range.x);
        __syncthreads();
        int p = base - 1 - threadIdx.x;
        if (threadIdx.x < cnt) {
            int id = point_list[p];
            s_id[threadIdx.x] = id;
            s_xy[threadIdx.x] = xy[id];
            s_co[threadIdx.x] = make_float4(conic[id * 3], conic[id * 3 + 1], conic[id * 3 + 2], opac[id]);
            s_col[threadIdx.x] = make_float3(color[id * 3], color[id * 3 + 1], color[id * 3 + 2]);
        }
        __syncthreads();
        for (int j = 0; j < cnt; ++j) {
            const int pos = base - 1 - j;
            float m[NM];
            float wsum = 0.f;
            bool contrib = false;
            if (pos < my_last) {
                float dx = s_xy[j].x - fx_, dy = s_xy[j].y - fy_;
                float4 co = s_co[j];
                float power = -0.5f * (co.x * dx * dx + co.z * dy * dy) - co.y * dx * dy;
                if (power <= 0.f) {
                    float g_raw = co.w * __expf(power);
                    float alpha = fminf(MAX_ALPHA, g_raw);
                    if (alpha >= MIN_ALPHA) {
                        float Ti = Tcur / (1.f - alpha);
                        float3 c = s_col[j];
                        float inv = 1.f / (1.f - alpha);
                        float d0 = c.x * Ti - S0 * inv;
                        float d1 = c.y * Ti - S1 * inv;
                        float d2 = c.z * Ti - S2 * inv;
                        float wgt = d0 * d0 + d1 * d1 + d2 * d2;
                        wsum = alpha * Ti;
                        if (g_raw < MAX_ALPHA) {
                            float g[NU];
                            g[0] = -alpha * (co.x * dx + co.y * dy);
                            g[1] = -alpha * (co.y * dx + co.z * dy);
                            g[2] = -0.5f * alpha * dx * dx;
                            g[3] = -alpha * dx * dy;
                            g[4] = -0.5f * alpha * dy * dy;
                            g[5] = alpha * (1.f - co.w);
                            int k = 0;
                            for (int a = 0; a < NU; ++a)
                                for (int b = a; b < NU; ++b) m[k++] = wgt * g[a] * g[b];
                            contrib = true;
                        }
                        S0 += c.x * alpha * Ti; S1 += c.y * alpha * Ti; S2 += c.z * alpha * Ti;
                        Tcur = Ti;
                    }
                }
            }
            if (!contrib) {
                for (int k = 0; k < NM; ++k) m[k] = 0.f;
            }
            unsigned any = __ballot_sync(0xffffffffu, contrib || wsum > 0.f);
            if (any == 0) continue;
            for (int off = 16; off > 0; off >>= 1) {
                for (int k = 0; k < NM; ++k) m[k] += __shfl_down_sync(0xffffffffu, m[k], off);
                wsum += __shfl_down_sync(0xffffffffu, wsum, off);
            }
            if (lane == 0) {
                const int id = s_id[j];
                for (int k = 0; k < NM; ++k)
                    if (m[k] != 0.f) atomicAdd(&M[(size_t)id * NM + k], m[k]);
                if (wsum > 0.f) atomicAdd(&Wsum[id], wsum);
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// 4. per-Gaussian: F += A^T M A, plus observation statistics. Resets M and Wsum.
// ---------------------------------------------------------------------------------
extern "C" __global__ void combine(
    const int N, const Cam* __restrict__ camp,
    const float* __restrict__ means, const float* __restrict__ scales, const float* __restrict__ quats,
    float* __restrict__ M, float* __restrict__ Wsum, float* __restrict__ F,
    float* __restrict__ obs /* per Gaussian: n_views, w_total, sum w*d (3), sum w*d d^T (6) = 11 */)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float wv = Wsum[i];
    if (wv <= 0.f) return;
    const Cam cam = *camp;
    double Mfull[NU][NU];
    {
        int k = 0;
        for (int a = 0; a < NU; ++a)
            for (int b = a; b < NU; ++b) { double v = M[(size_t)i * NM + k]; Mfull[a][b] = v; Mfull[b][a] = v; ++k; }
    }
    for (int k = 0; k < NM; ++k) M[(size_t)i * NM + k] = 0.f;
    Wsum[i] = 0.f;

    double th[9] = {means[i * 3], means[i * 3 + 1], means[i * 3 + 2],
                    fmin((double)scales[i * 3], 20.0), fmin((double)scales[i * 3 + 1], 20.0), fmin((double)scales[i * 3 + 2], 20.0),
                    0.0, 0.0, 0.0};
    double q[4] = {quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]};
    double A[NU][NP];
    for (int a = 0; a < NU; ++a)
        for (int b = 0; b < NP; ++b) A[a][b] = 0.0;
    double u0[5], z0;
    if (!project_u<double>(cam, th, q, u0, &z0)) return;
    // central differences, steps relative to the parameter's natural scale
    double smax = fmax(exp(th[3]), fmax(exp(th[4]), exp(th[5])));
    for (int p = 0; p < 9; ++p) {
        double h = p < 3 ? fmax(1e-4 * smax, 1e-6 * z0) : 1e-4;
        double tp[9], tm[9], up[5], um[5];
        for (int k = 0; k < 9; ++k) { tp[k] = th[k]; tm[k] = th[k]; }
        tp[p] += h; tm[p] -= h;
        bool okp = project_u<double>(cam, tp, q, up, nullptr);
        bool okm = project_u<double>(cam, tm, q, um, nullptr);
        if (okp && okm) for (int a = 0; a < 5; ++a) A[a][p] = (up[a] - um[a]) / (2.0 * h);
        else if (okp) for (int a = 0; a < 5; ++a) A[a][p] = (up[a] - u0[a]) / h;
        else if (okm) for (int a = 0; a < 5; ++a) A[a][p] = (u0[a] - um[a]) / h;
    }
    A[5][9] = 1.0;  // opacity logit maps straight through
    // B = M A (6 x 10), F += A^T B
    double B[NU][NP];
    for (int a = 0; a < NU; ++a)
        for (int b = 0; b < NP; ++b) {
            double s = 0.0;
            for (int k = 0; k < NU; ++k) s += Mfull[a][k] * A[k][b];
            B[a][b] = s;
        }
    int k = 0;
    for (int r = 0; r < NP; ++r)
        for (int c = r; c < NP; ++c) {
            double s = 0.0;
            for (int a = 0; a < NU; ++a) s += A[a][r] * B[a][c];
            F[(size_t)i * NF + k] += (float)s;
            ++k;
        }
    // observation statistics (blend-weighted viewing directions)
    double d[3] = {th[0] - cam.campos[0], th[1] - cam.campos[1], th[2] - cam.campos[2]};
    double dn = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) + 1e-20;
    for (int a = 0; a < 3; ++a) d[a] /= dn;
    float* o = obs + (size_t)i * 11;
    o[0] += 1.f;
    o[1] += wv;
    for (int a = 0; a < 3; ++a) o[2 + a] += (float)(wv * d[a]);
    int kk = 5;
    for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) o[kk++] += (float)(wv * d[a] * d[b]);
}
"""
