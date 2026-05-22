#include <cmath>

namespace pricing {
struct Greeks {
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

namespace {
constexpr double kEpsilon = 1.0e-12;
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr double kInvSqrt2Pi = 0.39894228040143267794;

double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

double normal_pdf(double x) {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}
}  // namespace

Greeks black_scholes_greeks(
    double spot,
    double strike,
    double time_to_maturity,
    double risk_free_rate,
    double volatility,
    int option_type
) {
    if (time_to_maturity <= kEpsilon || volatility <= kEpsilon) {
        double delta = 0.0;
        if (option_type >= 0) {
            delta = spot > strike ? 1.0 : spot < strike ? 0.0 : 0.5;
        } else {
            delta = spot < strike ? -1.0 : spot > strike ? 0.0 : -0.5;
        }
        return {delta, 0.0, 0.0, 0.0, 0.0};
    }

    const double sqrt_t = std::sqrt(time_to_maturity);
    const double variance = volatility * volatility;
    const double d1 = (
        std::log(spot / strike)
        + (risk_free_rate + 0.5 * variance) * time_to_maturity
    ) / (volatility * sqrt_t);
    const double d2 = d1 - volatility * sqrt_t;
    const double pdf_d1 = normal_pdf(d1);
    const double discount_factor = std::exp(-risk_free_rate * time_to_maturity);

    const double gamma = pdf_d1 / (spot * volatility * sqrt_t);
    const double vega = spot * pdf_d1 * sqrt_t;
    const double carry_term = risk_free_rate * strike * discount_factor;
    const double decay_term = -(spot * pdf_d1 * volatility) / (2.0 * sqrt_t);

    if (option_type >= 0) {
        const double delta = normal_cdf(d1);
        const double theta = decay_term - carry_term * normal_cdf(d2);
        const double rho = strike * time_to_maturity * discount_factor * normal_cdf(d2);
        return {delta, gamma, vega, theta, rho};
    }

    const double delta = normal_cdf(d1) - 1.0;
    const double theta = decay_term + carry_term * normal_cdf(-d2);
    const double rho = -strike * time_to_maturity * discount_factor * normal_cdf(-d2);
    return {delta, gamma, vega, theta, rho};
}
}  // namespace pricing
