#include <seal/seal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

using namespace seal;

int main() {
    // CKKS parameters
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384; // plenty for add + const-mul
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

    SEALContext context(parms);
    if (!context.parameters_set()) {
        std::cerr << "SEAL parameters are not valid!\n";
        return 1;
    }

    KeyGenerator keygen(context);
    auto sk = keygen.secret_key();
    PublicKey pk; keygen.create_public_key(pk);

    // Save params
    { std::ofstream ofs("parms.bin", std::ios::binary); parms.save(ofs); }
    // Save public key (share with hospitals + aggregator)
    { std::ofstream ofs("public.key", std::ios::binary); pk.save(ofs); }
    // Save secret key (KEEP PRIVATE with key owner)
    { std::ofstream ofs("secret.key", std::ios::binary); sk.save(ofs); }

    std::cout << "Generated: parms.bin, public.key, secret.key\n";
    return 0;
}
