/*
 * eps_star_ssim.cpp — Busca binária por ε* via fidelidade SSIM.
 *
 * Definição: ε* = inf{ε : SSIM(I, reconstrução(ε)) < threshold}
 *
 * Reconstrução em ε: cada pixel é substituído pela cor média (RGB)
 * do seu componente conexo no grafo de pixels filtrado por threshold
 * ε em distância Lab, conectividade 4.
 *
 * Reaproveita o union-find rápido do coarse2.cpp (O(N α(N))) e
 * implementa SSIM multicanal com janela gaussiana 11x11 — o mesmo
 * algoritmo usado por skimage.metrics.structural_similarity.
 *
 * Uso:
 *   ./eps_star_ssim image.jpg [--ssim_threshold 0.95]
 *                             [--eps_min 0.5] [--eps_max 50.0]
 *                             [--tol 0.5] [--verbose]
 *
 * Saída: imprime ε*, número de avaliações, tempo total (stdout)
 * e opcionalmente grava um .csv simples se --output for passado.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// ── Configuração ──────────────────────────────────────────────────────────────

static const int TARGET_SIDE = 1024;

// ── Conversão RGB → Lab (idêntico ao coarse2.cpp) ────────────────────────────

static void rgb_to_lab(float r, float g, float b,
                        float& L, float& A, float& B) {
    auto lin = [](float v) -> float {
        v /= 255.0f;
        return (v > 0.04045f) ? std::pow((v + 0.055f) / 1.055f, 2.4f)
                               : (v / 12.92f);
    };
    float vR = lin(r), vG = lin(g), vB = lin(b);
    float X = vR*0.4124f + vG*0.3576f + vB*0.1805f;
    float Y = vR*0.2126f + vG*0.7152f + vB*0.0722f;
    float Z = vR*0.0193f + vG*0.1192f + vB*0.9505f;
    X /= 0.95047f; Y /= 1.0f; Z /= 1.08883f;
    auto f = [](float t) -> float {
        return (t > 0.008856f) ? std::cbrt(t) : (7.787f*t + 16.0f/116.0f);
    };
    L = 116.0f*f(Y) - 16.0f;
    A = 500.0f*(f(X) - f(Y));
    B = 200.0f*(f(Y) - f(Z));
}

// ── Resize bilinear → mantém RGB E calcula Lab ───────────────────────────────

static void resize_rgb_and_lab(const uint8_t* src, int sw, int sh,
                                int dw, int dh,
                                std::vector<uint8_t>& out_rgb,
                                std::vector<float>& out_lab) {
    out_rgb.resize(dw * dh * 3);
    out_lab.resize(dw * dh * 3);
    float sx = (float)sw / dw, sy = (float)sh / dh;

    for (int dr = 0; dr < dh; ++dr) {
        float sr_f = (dr + 0.5f)*sy - 0.5f;
        int sr0 = std::max(0, (int)std::floor(sr_f));
        int sr1 = std::min(sh-1, sr0+1);
        float wy = sr_f - sr0;
        for (int dc = 0; dc < dw; ++dc) {
            float sc_f = (dc + 0.5f)*sx - 0.5f;
            int sc0 = std::max(0, (int)std::floor(sc_f));
            int sc1 = std::min(sw-1, sc0+1);
            float wx = sc_f - sc0;
            auto px = [&](int r, int c) -> const uint8_t* {
                return &src[(r*sw+c)*3];
            };
            float R = (1-wy)*((1-wx)*px(sr0,sc0)[0] + wx*px(sr0,sc1)[0])
                    +    wy *((1-wx)*px(sr1,sc0)[0] + wx*px(sr1,sc1)[0]);
            float G = (1-wy)*((1-wx)*px(sr0,sc0)[1] + wx*px(sr0,sc1)[1])
                    +    wy *((1-wx)*px(sr1,sc0)[1] + wx*px(sr1,sc1)[1]);
            float B = (1-wy)*((1-wx)*px(sr0,sc0)[2] + wx*px(sr0,sc1)[2])
                    +    wy *((1-wx)*px(sr1,sc0)[2] + wx*px(sr1,sc1)[2]);
            R = std::clamp(R, 0.f, 255.f);
            G = std::clamp(G, 0.f, 255.f);
            B = std::clamp(B, 0.f, 255.f);

            int idx = (dr*dw+dc)*3;
            out_rgb[idx]   = (uint8_t)std::round(R);
            out_rgb[idx+1] = (uint8_t)std::round(G);
            out_rgb[idx+2] = (uint8_t)std::round(B);

            float L, A, Bv;
            rgb_to_lab(R, G, B, L, A, Bv);
            out_lab[idx] = L; out_lab[idx+1] = A; out_lab[idx+2] = Bv;
        }
    }
}

// ── Union-Find com union-by-rank e path halving ───────────────────────────────

struct UF {
    std::vector<int>      parent;
    std::vector<int>      rank_;
    std::vector<uint32_t> size_;

    explicit UF(int n) : parent(n), rank_(n, 0), size_(n, 1) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        size_[a] += size_[b];
        if (rank_[a] == rank_[b]) rank_[a]++;
    }
};

// ── Arestas ordenadas — construídas uma vez por imagem ───────────────────────

struct Edge { float w; int u, v; };

static std::vector<Edge> build_sorted_edges(const std::vector<float>& lab,
                                             int W, int H) {
    std::vector<Edge> edges;
    edges.reserve(2 * W * H);

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int i = r*W+c;
            if (c < W-1) {
                int j = i+1;
                float dL = lab[i*3]   - lab[j*3];
                float da = lab[i*3+1] - lab[j*3+1];
                float db = lab[i*3+2] - lab[j*3+2];
                edges.push_back({std::sqrt(dL*dL+da*da+db*db), i, j});
            }
            if (r < H-1) {
                int j = i+W;
                float dL = lab[i*3]   - lab[j*3];
                float da = lab[i*3+1] - lab[j*3+1];
                float db = lab[i*3+2] - lab[j*3+2];
                edges.push_back({std::sqrt(dL*dL+da*da+db*db), i, j});
            }
        }
    }
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b){ return a.w < b.w; });
    return edges;
}

// ── Reconstrução: cada componente pintado pela cor RGB média ────────────────

static std::vector<uint8_t> reconstruct(const std::vector<uint8_t>& rgb,
                                         UF& uf, int npx) {
    std::vector<double> sumR(npx, 0), sumG(npx, 0), sumB(npx, 0);
    std::vector<uint32_t> cnt(npx, 0);

    for (int i = 0; i < npx; ++i) {
        int r = uf.find(i);
        sumR[r] += rgb[i*3];
        sumG[r] += rgb[i*3+1];
        sumB[r] += rgb[i*3+2];
        cnt[r]++;
    }

    std::vector<uint8_t> out(npx * 3);
    for (int i = 0; i < npx; ++i) {
        int r = uf.find(i);
        out[i*3]   = (uint8_t)std::round(sumR[r] / cnt[r]);
        out[i*3+1] = (uint8_t)std::round(sumG[r] / cnt[r]);
        out[i*3+2] = (uint8_t)std::round(sumB[r] / cnt[r]);
    }
    return out;
}

// ── SSIM multicanal com janela gaussiana 11x11 ───────────────────────────────

static std::vector<float> gaussian_kernel_1d(int size, float sigma) {
    std::vector<float> k(size);
    int half = size / 2;
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float x = (float)(i - half);
        k[i] = std::exp(-(x*x) / (2*sigma*sigma));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;
    return k;
}

static void gaussian_blur(const std::vector<float>& img, int W, int H,
                           const std::vector<float>& kernel,
                           std::vector<float>& out) {
    int half = (int)kernel.size() / 2;
    std::vector<float> tmp(W * H);

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            float acc = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int cc = std::clamp(c + k, 0, W - 1);
                acc += img[r*W + cc] * kernel[k + half];
            }
            tmp[r*W + c] = acc;
        }
    }
    out.resize(W * H);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            float acc = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int rr = std::clamp(r + k, 0, H - 1);
                acc += tmp[rr*W + c] * kernel[k + half];
            }
            out[r*W + c] = acc;
        }
    }
}

static double ssim_channel(const std::vector<float>& a,
                            const std::vector<float>& b,
                            int W, int H) {
    static const float C1 = (0.01f*255.0f) * (0.01f*255.0f);
    static const float C2 = (0.03f*255.0f) * (0.03f*255.0f);
    static const std::vector<float> kernel = gaussian_kernel_1d(11, 1.5f);

    std::vector<float> mu_a, mu_b;
    gaussian_blur(a, W, H, kernel, mu_a);
    gaussian_blur(b, W, H, kernel, mu_b);

    std::vector<float> a2(W*H), b2(W*H), ab(W*H);
    for (int i = 0; i < W*H; ++i) {
        a2[i] = a[i]*a[i];
        b2[i] = b[i]*b[i];
        ab[i] = a[i]*b[i];
    }
    std::vector<float> mu_a2, mu_b2, mu_ab;
    gaussian_blur(a2, W, H, kernel, mu_a2);
    gaussian_blur(b2, W, H, kernel, mu_b2);
    gaussian_blur(ab, W, H, kernel, mu_ab);

    double sum_ssim = 0.0;

    for (int i = 0; i < W*H; ++i) {
        float var_a  = mu_a2[i] - mu_a[i]*mu_a[i];
        float var_b  = mu_b2[i] - mu_b[i]*mu_b[i];
        float cov_ab = mu_ab[i] - mu_a[i]*mu_b[i];

        float num = (2*mu_a[i]*mu_b[i] + C1) * (2*cov_ab + C2);
        float den = (mu_a[i]*mu_a[i] + mu_b[i]*mu_b[i] + C1) * (var_a + var_b + C2);

        sum_ssim += (den > 1e-12f) ? (num/den) : 1.0;
    }
    return sum_ssim / (W*H);
}

static double ssim_rgb(const std::vector<uint8_t>& img_a,
                        const std::vector<uint8_t>& img_b,
                        int W, int H) {
    std::vector<float> ch_a(W*H), ch_b(W*H);
    double total = 0.0;

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < W*H; ++i) {
            ch_a[i] = (float)img_a[i*3+c];
            ch_b[i] = (float)img_b[i*3+c];
        }
        total += ssim_channel(ch_a, ch_b, W, H);
    }
    return total / 3.0;
}

// ── Avaliação de fidelidade em um dado ε ─────────────────────────────────────

static double fidelity_at_epsilon(const std::vector<Edge>& sorted_edges,
                                   const std::vector<uint8_t>& rgb,
                                   int W, int H, float epsilon) {
    int npx = W * H;
    UF uf(npx);
    for (const auto& e : sorted_edges) {
        if (e.w > epsilon) break;
        uf.unite(e.u, e.v);
    }
    std::vector<uint8_t> recon = reconstruct(rgb, uf, npx);
    return ssim_rgb(rgb, recon, W, H);
}

// ── Busca binária por ε* ──────────────────────────────────────────────────────

struct SearchResult {
    float  eps_star;
    int    n_evaluations;
    double elapsed_sec;
    std::vector<std::pair<float,double>> history;
};

static SearchResult find_eps_star(const std::vector<Edge>& sorted_edges,
                                   const std::vector<uint8_t>& rgb,
                                   int W, int H,
                                   double ssim_threshold,
                                   float eps_min, float eps_max,
                                   float tol, bool verbose) {
    auto t0 = std::chrono::high_resolution_clock::now();
    SearchResult res{};
    res.n_evaluations = 0;

    auto eval = [&](float eps) -> double {
        double s = fidelity_at_epsilon(sorted_edges, rgb, W, H, eps);
        res.history.push_back({eps, s});
        res.n_evaluations++;
        if (verbose) {
            std::cout << "    eps=" << eps << "  ssim=" << s << "\n";
        }
        return s;
    };

    double s_min = eval(eps_min);
    double s_max = eval(eps_max);

    if (s_min < ssim_threshold) {
        res.eps_star = eps_min;
    } else if (s_max >= ssim_threshold) {
        res.eps_star = eps_max;
    } else {
        float lo = eps_min, hi = eps_max;
        while ((hi - lo) > tol) {
            float mid = (lo + hi) / 2.0f;
            double s = eval(mid);
            if (s >= ssim_threshold) lo = mid;
            else                     hi = mid;
        }
        res.eps_star = lo;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    return res;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " image.jpg "
                  << "[--ssim_threshold 0.95] [--eps_min 0.5] "
                  << "[--eps_max 50.0] [--tol 0.5] [--target_side 1024] "
                  << "[--verbose] [--output result.csv]\n";
        return 1;
    }

    std::string img_path = argv[1];
    double ssim_threshold = 0.95;
    float  eps_min = 0.5f, eps_max = 50.0f, tol = 0.5f;
    int    target_side = TARGET_SIDE;
    bool   verbose = false;
    std::string output_path;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ssim_threshold" && i+1 < argc) ssim_threshold = std::stod(argv[++i]);
        else if (a == "--eps_min"   && i+1 < argc) eps_min = std::stof(argv[++i]);
        else if (a == "--eps_max"   && i+1 < argc) eps_max = std::stof(argv[++i]);
        else if (a == "--tol"       && i+1 < argc) tol = std::stof(argv[++i]);
        else if (a == "--target_side" && i+1 < argc) target_side = std::stoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else if (a == "--output" && i+1 < argc) output_path = argv[++i];
    }

    auto t_load0 = std::chrono::high_resolution_clock::now();

    int w, h, ch;
    uint8_t* raw = stbi_load(img_path.c_str(), &w, &h, &ch, 3);
    if (!raw) {
        std::cerr << "Erro ao carregar: " << img_path << "\n";
        return 1;
    }

    int nw = w, nh = h;
    if (std::max(w,h) > target_side) {
        float sc = (float)target_side / std::max(w,h);
        nw = std::max(1, (int)(w*sc));
        nh = std::max(1, (int)(h*sc));
    }

    std::vector<uint8_t> rgb;
    std::vector<float>   lab;
    resize_rgb_and_lab(raw, w, h, nw, nh, rgb, lab);
    stbi_image_free(raw);

    std::vector<Edge> sorted_edges = build_sorted_edges(lab, nw, nh);

    auto t_load1 = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(t_load1 - t_load0).count();

    if (verbose) {
        std::cout << "Imagem: " << nw << "x" << nh
                  << "  (" << (nw*nh) << " pixels)\n";
        std::cout << "Carregamento + ordenacao de arestas: "
                  << load_time << "s\n\n";
        std::cout << "Busca binaria por eps* (ssim < " << ssim_threshold << "):\n";
    }

    SearchResult result = find_eps_star(
        sorted_edges, rgb, nw, nh,
        ssim_threshold, eps_min, eps_max, tol, verbose);

    std::cout << "eps_star=" << result.eps_star
              << " n_evaluations=" << result.n_evaluations
              << " elapsed_sec=" << result.elapsed_sec
              << " load_sec=" << load_time
              << " W=" << nw << " H=" << nh
              << "\n";

    if (!output_path.empty()) {
        std::ofstream out(output_path);
        out << "epsilon,ssim\n";
        for (auto& [eps, s] : result.history) {
            out << eps << "," << s << "\n";
        }
        out.close();
    }

    return 0;
}
