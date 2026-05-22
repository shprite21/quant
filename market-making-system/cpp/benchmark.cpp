#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace pricing {
double black_scholes_price(
    double spot,
    double strike,
    double time_to_maturity,
    double risk_free_rate,
    double volatility,
    int option_type
);
}  // namespace pricing

int main(int argc, char** argv) {
    int n_options = 200000;
    std::uint32_t seed = 123;
    if (argc >= 2) {
        n_options = std::max(1, std::atoi(argv[1]));
    }
    if (argc >= 3) {
        seed = static_cast<std::uint32_t>(std::atoi(argv[2]));
    }

    std::mt19937 rng(seed);
    std::lognormal_distribution<double> spot_dist(std::log(100.0), 0.15);
    std::uniform_real_distribution<double> strike_dist(75.0, 125.0);
    std::uniform_real_distribution<double> maturity_dist(1.0 / 252.0, 1.0);
    std::uniform_real_distribution<double> rate_dist(0.0, 0.05);
    std::uniform_real_distribution<double> vol_dist(0.10, 0.55);
    std::bernoulli_distribution type_dist(0.5);

    std::vector<double> spots;
    std::vector<double> strikes;
    std::vector<double> maturities;
    std::vector<double> rates;
    std::vector<double> vols;
    std::vector<int> option_types;
    spots.reserve(n_options);
    strikes.reserve(n_options);
    maturities.reserve(n_options);
    rates.reserve(n_options);
    vols.reserve(n_options);
    option_types.reserve(n_options);

    for (int i = 0; i < n_options; ++i) {
        spots.push_back(spot_dist(rng));
        strikes.push_back(strike_dist(rng));
        maturities.push_back(maturity_dist(rng));
        rates.push_back(rate_dist(rng));
        vols.push_back(vol_dist(rng));
        option_types.push_back(type_dist(rng) ? 1 : -1);
    }

    double checksum = 0.0;
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_options; ++i) {
        checksum += pricing::black_scholes_price(
            spots[i],
            strikes[i],
            maturities[i],
            rates[i],
            vols[i],
            option_types[i]
        );
    }
    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    const double elapsed_seconds = elapsed.count();
    const double options_per_second = elapsed_seconds > 0.0 ? n_options / elapsed_seconds : 0.0;

    std::cout << std::fixed << std::setprecision(10)
              << "{"
              << "\"engine\":\"cpp\","
              << "\"options\":" << n_options << ","
              << "\"elapsed_seconds\":" << elapsed_seconds << ","
              << "\"options_per_second\":" << options_per_second << ","
              << "\"checksum\":" << checksum
              << "}";
    return 0;
}
