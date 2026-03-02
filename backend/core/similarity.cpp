#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;
using namespace std;

float calculate_cosine(const vector<float>& v1, const vector<float>& v2) {
    float dot_product = 0.0, norm_v1 = 0.0, norm_v2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }
    if (norm_v1 == 0 || norm_v2 == 0) return 0.0f;
    return dot_product / (sqrt(norm_v1) * sqrt(norm_v2));
}

vector<float> batch_similarity(const vector<float>& query_vec, const vector<vector<float>>& doc_vecs) {
    vector<float> scores;
    scores.reserve(doc_vecs.size());
    for (const auto& doc_vec : doc_vecs) {
        scores.push_back(calculate_cosine(query_vec, doc_vec));
    }
    return scores;
}

PYBIND11_MODULE(neutron_math, m) {
    m.def("batch_similarity", &batch_similarity, "Calculate cosine similarity between a query and multiple documents");
}