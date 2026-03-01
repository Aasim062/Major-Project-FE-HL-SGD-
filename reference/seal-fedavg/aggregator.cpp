#include <seal/seal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

using namespace seal;

// Load count from "<ct_path>.count.txt" if present; else NaN
double maybe_load_count(const std::string &ct_path) {
    std::ifstream ifs(ct_path + ".count.txt");
    if (!ifs) return std::numeric_limits<double>::quiet_NaN();
    double v; ifs >> v; return v;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage:\n  aggregator <parms.bin> <output_agg.ct> <mode: simple|weighted> <ct1> <ct2> [ct3 ...]\n";
        return 1;
    }

    std::string parms_path = argv[1];
    std::string out_path   = argv[2];
    std::string mode       = argv[3];
    std::vector<std::string> ct_paths;
    for (int i = 4; i < argc; ++i) ct_paths.emplace_back(argv[i]);

    bool weighted = (mode == "weighted");

    // Load parms & context
    seal::EncryptionParameters parms;
    {
        std::ifstream ifs(parms_path, std::ios::binary);
        if (!ifs) { std::cerr << "Failed to open parms.bin\n"; return 1; }
        parms.load(ifs);
    }
    auto context = seal::SEALContext(parms);
    if (!context.parameters_set()) {
        std::cerr << "SEAL parameters are not valid!\n"; return 1;
    }

    seal::Evaluator evaluator(context);

    // Load ciphertexts
    std::vector<seal::Ciphertext> cts(ct_paths.size());
    for (size_t i = 0; i < ct_paths.size(); ++i) {
        std::ifstream ifs(ct_paths[i], std::ios::binary);
        if (!ifs) { std::cerr << "Failed to open " << ct_paths[i] << "\n"; return 1; }
        cts[i].load(context, ifs);
    }

    seal::Ciphertext agg;
    bool agg_initialized = false;

    if (!weighted) {
        // Simple mean: (1/N) * sum(ct_i)
        for (size_t i = 0; i < cts.size(); ++i) {
            if (!agg_initialized) { agg = cts[i]; agg_initialized = true; }
            else { evaluator.add_inplace(agg, cts[i]); }
        }
        double invN = 1.0 / static_cast<double>(cts.size());
        evaluator.multiply_const_inplace(agg, invN);
    } else {
        // Weighted mean using plaintext counts stored alongside each ct
        std::vector<double> counts(cts.size());
        double total = 0.0;
        for (size_t i = 0; i < cts.size(); ++i) {
            double ci = maybe_load_count(ct_paths[i]);
            if (std::isnan(ci) || ci <= 0.0) {
                std::cerr << "Missing/invalid count for " << ct_paths[i]
                          << " (expected in " << ct_paths[i] << ".count.txt)\n";
                return 1;
            }
            counts[i] = ci; total += ci;
        }
        if (total <= 0.0) { std::cerr << "Total count must be positive.\n"; return 1; }

        for (size_t i = 0; i < cts.size(); ++i) {
            seal::Ciphertext term = cts[i];
            evaluator.multiply_const_inplace(term, counts[i] / total);
            if (!agg_initialized) { agg = term; agg_initialized = true; }
            else { evaluator.add_inplace(agg, term); }
        }
    }

    // Save aggregated ciphertext
    {
        std::ofstream ofs(out_path, std::ios::binary);
        if (!ofs) { std::cerr << "Cannot open " << out_path << " for writing.\n"; return 1; }
        agg.save(ofs);
    }

    std::cout << "Wrote aggregated ciphertext to " << out_path << "\n";
    return 0;
}
