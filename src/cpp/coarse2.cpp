/*
 * coarse2.cpp — Análise de componentes conexos cromáticos por threshold.
 *
 * Substitui o HEM por análise de percolação:
 *   - Para cada ε em lista de thresholds, calcula componentes conexos
 *     via union-find sobre pixels com conectividade 4
 *   - Processa todos os thresholds em uma única passagem sobre as arestas
 *     (ordenadas por peso) — O(E log E) total
 *   - Modo "features": salva curva G(ε), N(ε), P(s,ε) por threshold
 *   - Modo "graph":    idem + grafo RAG materializado em ε*
 *
 * Uso:
 *   ./coarse2 image.jpg out.coarse [--graph] [--epsilon_star 10.0]
 *             [--window x0 y0 x1 y1]   (frações 0-1; default = imagem inteira)
 *
 * Formato de saída .coarse:
 *   [uint32 mode]          0=features, 1=graph
 *   [uint32 n_thresholds]
 *   [float  epsilon_star]  (só se mode=1)
 *   [PercolStep * n_thresholds]
 *   -- se mode=1 --
 *   [uint32 n_nodes]
 *   [NodeData * n_nodes]
 *   [uint32 n_edges]
 *   [EdgeData * n_edges]
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ── Configuração ──────────────────────────────────────────────────────────────

static const int TARGET_SIDE = 1024;

// Thresholds padrão — escala aproximadamente geométrica em unidades Lab
static const std::vector<float> DEFAULT_EPSILONS = {
    2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 15.0f, 18.0f, 22.0f, 27.0f, 33.0f, 40.0f
};

// ── Estruturas de saída ───────────────────────────────────────────────────────

#pragma pack(push, 1)

struct PercolStep {
    float    epsilon;
    uint32_t n_components;   // número de componentes conexos
    uint32_t giant_size;     // tamanho do maior componente (pixels)
    float    giant_frac;     // giant_size / n_pixels
    float    entropy;        // entropia de Shannon da distribuição P(s)
    float    mean_size;      // tamanho médio dos componentes
    float    gini;           // desigualdade da distribuição de tamanhos
    float    n_singletons_frac; // fração de componentes de tamanho 1
};

struct NodeData {
    float L, a, b;    // cor Lab média (ponderada por área)
    float X, Y;       // posição normalizada [0,1]
    float area_frac;  // fração de pixels no componente
    uint32_t area_px; // tamanho em pixels
};

struct EdgeData {
    uint32_t src;
    uint32_t dst;
    float    color_dist;    // distância Lab mínima entre os componentes
    float    spatial_dist;  // distância espacial normalizada entre centróides
};

#pragma pack(pop)

// ── Conversão RGB → Lab ───────────────────────────────────────────────────────

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

// ── Resize bilinear → Lab ─────────────────────────────────────────────────────

static std::vector<float> resize_to_lab(const uint8_t* src,
                                         int sw, int sh,
                                         int dw, int dh) {
    std::vector<float> out(dw * dh * 3);
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
            float L, A, Bv;
            rgb_to_lab(std::clamp(R,0.f,255.f),
                       std::clamp(G,0.f,255.f),
                       std::clamp(B,0.f,255.f), L, A, Bv);
            int idx = (dr*dw+dc)*3;
            out[idx] = L; out[idx+1] = A; out[idx+2] = Bv;
        }
    }
    return out;
}

// ── Union-Find com path compression e union by rank ───────────────────────────

struct UF {
    std::vector<int>      parent;
    std::vector<int>      rank_;
    std::vector<uint32_t> size_;
    int n_components;

    explicit UF(int n) : parent(n), rank_(n, 0), size_(n, 1),
                          n_components(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    }

    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        size_[a] += size_[b];
        if (rank_[a] == rank_[b]) rank_[a]++;
        n_components--;
        return true;
    }

    uint32_t comp_size(int x) { return size_[find(x)]; }
};

// ── Calcular métricas de percolação ───────────────────────────────────────────

static PercolStep calc_metrics(const UF& uf, float epsilon, int npx) {
    // Coletar tamanhos únicos (um por componente raiz)
    std::vector<uint32_t> sizes;
    sizes.reserve(uf.n_components);
    for (int i = 0; i < npx; ++i) {
        if (uf.parent[i] == i) {  // é raiz
            sizes.push_back(uf.size_[i]);
        }
    }

    uint32_t giant = *std::max_element(sizes.begin(), sizes.end());
    uint64_t total = 0;
    for (auto s : sizes) total += s;
    double mean = (double)total / sizes.size();

    // Entropia de Shannon
    double ent = 0.0;
    for (auto s : sizes) {
        double p = (double)s / npx;
        if (p > 0) ent -= p * std::log2(p);
    }

    // Gini
    std::sort(sizes.begin(), sizes.end());
    double gini = 0.0;
    if (!sizes.empty() && total > 0) {
        double cum = 0;
        int n = sizes.size();
        for (int i = 0; i < n; ++i) {
            cum += sizes[i];
            gini += (double)sizes[i] * (2*(i+1) - n - 1);
        }
        gini /= ((double)n * total);
    }

    uint32_t singletons = 0;
    for (auto s : sizes) if (s == 1) singletons++;

    PercolStep ps{};
    ps.epsilon           = epsilon;
    ps.n_components      = (uint32_t)sizes.size();
    ps.giant_size        = giant;
    ps.giant_frac        = (float)giant / npx;
    ps.entropy           = (float)ent;
    ps.mean_size         = (float)mean;
    ps.gini              = (float)gini;
    ps.n_singletons_frac = (float)singletons / sizes.size();
    return ps;
}

// ── Extrair grafo RAG em ε* ───────────────────────────────────────────────────

// wc0,wr0 = canto superior-esquerdo da janela em pixels
// win_w, win_h = dimensões da janela; nw = largura da imagem completa
// UF usa índices locais 0..win_w*win_h-1
static void extract_graph(const UF& uf,
                           const std::vector<float>& lab,
                           int wc0, int wr0, int win_w, int win_h, int nw,
                           std::vector<NodeData>& out_nodes,
                           std::vector<EdgeData>& out_edges) {
    int win_npx = win_w * win_h;

    // Mapear raiz local → id compacto
    std::vector<int> compact(win_npx, -1);
    int n_nodes = 0;
    UF& uf_m = const_cast<UF&>(uf);
    for (int i = 0; i < win_npx; ++i) {
        int r = uf_m.find(i);
        if (compact[r] < 0) compact[r] = n_nodes++;
    }

    // Acumular cor e posição por componente
    std::vector<double> sum_L(n_nodes,0), sum_a(n_nodes,0), sum_b(n_nodes,0);
    std::vector<double> sum_X(n_nodes,0), sum_Y(n_nodes,0);
    std::vector<uint32_t> px_count(n_nodes, 0);

    for (int lr = 0; lr < win_h; ++lr) {
        for (int lc = 0; lc < win_w; ++lc) {
            int li  = lr*win_w+lc;                  // índice local no UF
            int gi  = (lr+wr0)*nw+(lc+wc0);        // índice global no lab[]
            int cid = compact[uf_m.find(li)];
            sum_L[cid] += lab[gi*3];
            sum_a[cid] += lab[gi*3+1];
            sum_b[cid] += lab[gi*3+2];
            sum_X[cid] += lc;                       // coordenada local
            sum_Y[cid] += lr;
            px_count[cid]++;
        }
    }

    out_nodes.resize(n_nodes);
    for (int k = 0; k < n_nodes; ++k) {
        double cnt = px_count[k];
        out_nodes[k].L        = (float)(sum_L[k]/cnt);
        out_nodes[k].a        = (float)(sum_a[k]/cnt);
        out_nodes[k].b        = (float)(sum_b[k]/cnt);
        // X, Y normalizados para [0,1] dentro da janela
        out_nodes[k].X        = (win_w > 1) ? (float)(sum_X[k]/cnt / (win_w-1)) : 0.5f;
        out_nodes[k].Y        = (win_h > 1) ? (float)(sum_Y[k]/cnt / (win_h-1)) : 0.5f;
        out_nodes[k].area_frac = (float)cnt / win_npx;
        out_nodes[k].area_px  = px_count[k];
    }

    // Arestas RAG — varrer fronteiras horizontais e verticais dentro da janela
    std::vector<std::vector<std::pair<int,float>>> adj(n_nodes);

    // lu, lv = índices locais no UF; gu, gv = índices globais para lab[]
    auto add_edge = [&](int lu, int lv, int gu, int gv) {
        int cu = compact[uf_m.find(lu)];
        int cv = compact[uf_m.find(lv)];
        if (cu == cv) return;
        if (cu > cv) std::swap(cu, cv);
        float dL = lab[gu*3]   - lab[gv*3];
        float da = lab[gu*3+1] - lab[gv*3+1];
        float db = lab[gu*3+2] - lab[gv*3+2];
        adj[cu].push_back({cv, std::sqrt(dL*dL+da*da+db*db)});
    };

    for (int lr = 0; lr < win_h; ++lr)
        for (int lc = 0; lc < win_w-1; ++lc)
            add_edge(lr*win_w+lc,   lr*win_w+lc+1,
                     (lr+wr0)*nw+(lc+wc0), (lr+wr0)*nw+(lc+wc0+1));
    for (int lr = 0; lr < win_h-1; ++lr)
        for (int lc = 0; lc < win_w; ++lc)
            add_edge(lr*win_w+lc,   (lr+1)*win_w+lc,
                     (lr+wr0)*nw+(lc+wc0), (lr+1+wr0)*nw+(lc+wc0));

    // Deduplicar — manter mínimo cd por par
    out_edges.clear();
    for (int u = 0; u < n_nodes; ++u) {
        std::sort(adj[u].begin(), adj[u].end());
        for (int k = 0; k < (int)adj[u].size(); ) {
            int v = adj[u][k].first;
            float min_cd = adj[u][k].second;
            int k2 = k+1;
            while (k2 < (int)adj[u].size() && adj[u][k2].first == v) {
                min_cd = std::min(min_cd, adj[u][k2].second);
                k2++;
            }
            // Distância espacial entre centróides
            float dx = out_nodes[u].X - out_nodes[v].X;
            float dy = out_nodes[u].Y - out_nodes[v].Y;
            float sd = std::sqrt(dx*dx + dy*dy);

            EdgeData ed{};
            ed.src          = (uint32_t)u;
            ed.dst          = (uint32_t)v;
            ed.color_dist   = min_cd;
            ed.spatial_dist = sd;
            out_edges.push_back(ed);
            k = k2;
        }
    }
}

// ── Pipeline principal ────────────────────────────────────────────────────────

// wx0,wy0,wx1,wy1 — janela em frações [0,1]; default = imagem inteira.
// Suporta overlap: e.g. wx0=0, wx1=0.6 e wx0=0.4, wx1=1.0 para 2×2 com 20% overlap.
void process(const std::string& img_path,
             const std::string& out_path,
             const std::vector<float>& epsilons,
             bool save_graph,
             float epsilon_star,
             float wx0 = 0.0f, float wy0 = 0.0f,
             float wx1 = 1.0f, float wy1 = 1.0f) {

    // Carregar imagem
    int w, h, ch;
    uint8_t* raw = stbi_load(img_path.c_str(), &w, &h, &ch, 3);
    if (!raw) {
        std::cerr << "Erro ao carregar: " << img_path << "\n";
        return;
    }

    // Resize mantendo proporção
    int nw = w, nh = h;
    if (std::max(w,h) > TARGET_SIDE) {
        float sc = (float)TARGET_SIDE / std::max(w,h);
        nw = std::max(1, (int)(w*sc));
        nh = std::max(1, (int)(h*sc));
    }
    std::vector<float> lab = resize_to_lab(raw, w, h, nw, nh);
    stbi_image_free(raw);

    // Janela em pixels — clamped para dimensões reais
    int wc0 = std::max(0, std::min(nw-1, (int)(wx0 * nw)));
    int wc1 = std::max(wc0+1, std::min(nw,   (int)(wx1 * nw)));
    int wr0 = std::max(0, std::min(nh-1, (int)(wy0 * nh)));
    int wr1 = std::max(wr0+1, std::min(nh,   (int)(wy1 * nh)));
    int win_w   = wc1 - wc0;
    int win_h   = wr1 - wr0;
    int win_npx = win_w * win_h;

    // Índice local dentro da janela (0 .. win_npx-1)
    auto local_idx = [&](int r, int c) -> int {
        return (r - wr0) * win_w + (c - wc0);
    };

    // Construir arestas da grade 4-conectada restritas à janela
    struct Edge { float w; int u, v; };
    std::vector<Edge> edges;
    edges.reserve(2 * win_npx);

    for (int r = wr0; r < wr1; ++r) {
        for (int c = wc0; c < wc1; ++c) {
            int i  = local_idx(r, c);
            int gi = r*nw+c;   // índice global no lab[]
            // Aresta horizontal
            if (c < wc1-1) {
                int j  = local_idx(r, c+1);
                int gj = r*nw+c+1;
                float dL = lab[gi*3]   - lab[gj*3];
                float da = lab[gi*3+1] - lab[gj*3+1];
                float db = lab[gi*3+2] - lab[gj*3+2];
                edges.push_back({std::sqrt(dL*dL+da*da+db*db), i, j});
            }
            // Aresta vertical
            if (r < wr1-1) {
                int j  = local_idx(r+1, c);
                int gj = (r+1)*nw+c;
                float dL = lab[gi*3]   - lab[gj*3];
                float da = lab[gi*3+1] - lab[gj*3+1];
                float db = lab[gi*3+2] - lab[gj*3+2];
                edges.push_back({std::sqrt(dL*dL+da*da+db*db), i, j});
            }
        }
    }

    // Ordenar arestas por peso — O(E log E) uma vez
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b){ return a.w < b.w; });

    // Processar todos os thresholds em uma passagem
    // Varrer arestas em ordem crescente de peso,
    // adicionando ao UF quando peso <= threshold atual
    std::vector<float> eps_sorted = epsilons;
    std::sort(eps_sorted.begin(), eps_sorted.end());

    std::vector<PercolStep> steps;
    steps.reserve(eps_sorted.size());

    UF uf(win_npx);
    UF* uf_star = nullptr;  // snapshot em epsilon_star

    size_t edge_idx = 0;
    for (float eps : eps_sorted) {
        while (edge_idx < edges.size() && edges[edge_idx].w <= eps) {
            uf.unite(edges[edge_idx].u, edges[edge_idx].v);
            edge_idx++;
        }

        steps.push_back(calc_metrics(uf, eps, win_npx));

        if (save_graph && std::abs(eps - epsilon_star) < 1e-4f && !uf_star) {
            uf_star = new UF(uf);
        }
    }

    if (save_graph && !uf_star) {
        UF uf2(win_npx);
        for (auto& e : edges) {
            if (e.w > epsilon_star) break;
            uf2.unite(e.u, e.v);
        }
        uf_star = new UF(std::move(uf2));
    }

    // Extrair grafo se necessário
    std::vector<NodeData> nodes;
    std::vector<EdgeData> graph_edges;
    if (save_graph && uf_star) {
        extract_graph(*uf_star, lab, wc0, wr0, win_w, win_h, nw, nodes, graph_edges);
        delete uf_star;
    }

    // ── Gravar binário ────────────────────────────────────────────────────────
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::cerr << "Erro ao gravar: " << out_path << "\n";
        return;
    }

    uint32_t mode        = save_graph ? 1 : 0;
    uint32_t n_thresholds = (uint32_t)steps.size();

    out.write(reinterpret_cast<char*>(&mode),         sizeof(mode));
    out.write(reinterpret_cast<char*>(&n_thresholds), sizeof(n_thresholds));

    if (save_graph) {
        out.write(reinterpret_cast<const char*>(&epsilon_star), sizeof(epsilon_star));
    }

    out.write(reinterpret_cast<char*>(steps.data()),
               n_thresholds * sizeof(PercolStep));

    if (save_graph) {
        uint32_t nn = (uint32_t)nodes.size();
        uint32_t ne = (uint32_t)graph_edges.size();
        out.write(reinterpret_cast<char*>(&nn), sizeof(nn));
        out.write(reinterpret_cast<char*>(nodes.data()), nn * sizeof(NodeData));
        out.write(reinterpret_cast<char*>(&ne), sizeof(ne));
        out.write(reinterpret_cast<char*>(graph_edges.data()), ne * sizeof(EdgeData));
    }

    out.close();

    std::cout << "OK: " << img_path
              << "  mode=" << (save_graph ? "graph" : "features")
              << "  thresholds=" << n_thresholds
              << "  giant@eps" << epsilon_star << "="
              << (steps.empty() ? 0.0f : steps.back().giant_frac)
              << (save_graph ? "  nodes=" + std::to_string(nodes.size()) +
                               "  edges=" + std::to_string(graph_edges.size()) : "")
              << "\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0]
                  << " image.jpg out.coarse [--graph] [--epsilon_star N]\n";
        return 1;
    }

    std::string img_path  = argv[1];
    std::string out_path  = argv[2];
    bool        save_graph   = false;
    float       epsilon_star = 10.0f;
    float       wx0 = 0.0f, wy0 = 0.0f, wx1 = 1.0f, wy1 = 1.0f;

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--graph") {
            save_graph = true;
        } else if (std::string(argv[i]) == "--epsilon_star" && i+1 < argc) {
            epsilon_star = std::stof(argv[++i]);
        } else if (std::string(argv[i]) == "--window" && i+4 < argc) {
            wx0 = std::stof(argv[++i]);
            wy0 = std::stof(argv[++i]);
            wx1 = std::stof(argv[++i]);
            wy1 = std::stof(argv[++i]);
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    process(img_path, out_path, DEFAULT_EPSILONS, save_graph, epsilon_star,
            wx0, wy0, wx1, wy1);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Concluído em "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()
              << " ms\n";
    return 0;
}
