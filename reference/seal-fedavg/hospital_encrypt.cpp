#include <seal/seal.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

using namespace seal;

// Read a CSV: w1,w2,w3,w4,intercept[,count]
bool read_csv(const std::string &path, std::vector<double> &weights, double &count, bool &has_count) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::string header, line;
    std::getline(ifs, header);
    std::getline(ifs, line);
    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> cells;
    while (std::getline(ss, cell, ',')) cells.push_back(cell);

    if (cells.size() < 5) return false;
    weights.resize(5);
    for (int i = 0; i < 5; ++i) weights[i] = std::stod(cells[i]);

    if (cells.size() >= 6 && !cells[5].empty()) {
        has_count = true;
        count = std::stod(cells[5]);
    } else {
        has_count = false;
        count = 0.0;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage:\n  hospital_encrypt <parms.bin> <public.key> <input.csv> <output.ct>\n";
        return 1;
    }

    std::string parms_path = argv[1];
    std::string pub_path   = argv[2];
    std::string in_path    = argv[3];
    std::string out_ct     = argv[4];

    // Load parms & context
    EncryptionParameters parms;
    {
        std::ifstream ifs(parms_path, std::ios::binary);
        parms.load(ifs);
    }
    auto context = SEALContext(parms);

    // Load public key
    PublicKey pk;
    {
        std::ifstream ifs(pub_path, std::ios::binary);
        pk.load(context, ifs);
    }

    std::vector<double> w(5);
    double count = 0.0;
    bool has_count = false;
    if (!read_csv(in_path, w, count, has_count)) {
        std::cerr << "Failed to read input CSV.\n";
        return 1;
    }

    CKKSEncoder encoder(context);
    Encryptor encryptor(context, pk);

    double scale = std::pow(2.0, 40);
    Plaintext pt;
    encoder.encode(w, scale, pt);
    Ciphertext ct;
    encryptor.encrypt(pt, ct);

    // Save ciphertext
    {
        std::ofstream ofs(out_ct, std::ios::binary);
        ct.save(ofs);
    }

    if (has_count) {
        std::ofstream ofs(out_ct + ".count.txt");
        ofs << std::fixed << count << "\n";
    }

    std::cout << "Encrypted weights written to " << out_ct << "\n";
    return 0;
}
