#include <seal/seal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace seal;

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage:\n  decrypt <parms.bin> <secret.key> <agg.ct> <out.csv>\n";
        return 1;
    }
    std::string parms_path = argv[1];
    std::string sk_path    = argv[2];
    std::string ct_path    = argv[3];
    std::string out_csv    = argv[4];

    // Load parms/context
    EncryptionParameters parms;
    {
        std::ifstream ifs(parms_path, std::ios::binary);
        if (!ifs) { std::cerr << "Failed to open parms.bin\n"; return 1; }
        parms.load(ifs);
    }
    SEALContext context(parms);
    if (!context.parameters_set()) {
        std::cerr << "SEAL parameters are not valid!\n"; return 1;
    }

    // Load secret key
    SecretKey sk;
    {
        std::ifstream ifs(sk_path, std::ios::binary);
        if (!ifs) { std::cerr << "Failed to open secret.key\n"; return 1; }
        sk.load(context, ifs);
    }
    Decryptor decryptor(context, sk);
    CKKSEncoder encoder(context);

    // Load aggregated ciphertext
    Ciphertext ct;
    {
        std::ifstream ifs(ct_path, std::ios::binary);
        if (!ifs) { std::cerr << "Failed to open agg.ct\n"; return 1; }
        ct.load(context, ifs);
    }

    // Decrypt & decode
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    std::vector<double> decoded;
    encoder.decode(pt, decoded);

    if (decoded.size() < 5) {
        std::cerr << "Decoded vector too small.\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Aggregated (decrypted) weights:\n";
    std::cout << "w1: " << decoded[0] << "\n";
    std::cout << "w2: " << decoded[1] << "\n";
    std::cout << "w3: " << decoded[2] << "\n";
    std::cout << "w4: " << decoded[3] << "\n";
    std::cout << "b : " << decoded[4] << "\n";

    std::ofstream ofs(out_csv);
    ofs << "w1,w2,w3,w4,intercept\n";
    ofs << std::setprecision(16)
        << decoded[0] << "," << decoded[1] << "," << decoded[2] << ","
        << decoded[3] << "," << decoded[4] << "\n";

    std::cout << "Saved to " << out_csv << "\n";
    return 0;
}
