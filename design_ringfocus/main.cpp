#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>

#include <fftw3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <grid/bundle.hpp>
#include <grid/core.hpp>
#include <grid/linear.hpp>
#include <vector>

#include "constants.hpp"

auto reflection()
{


    return Eigen::Vector3d{};
}

#define LOG_CONST(x) std::cout << #x ": " << x << std::endl;

int main()
{

    LOG_CONST(f);
    LOG_CONST(L1);
    LOG_CONST(L2);
    LOG_CONST(ML);
    LOG_CONST(f1);
    LOG_CONST(f2);
    LOG_CONST(f3);
    LOG_CONST(f4);
    LOG_CONST(phi);
    LOG_CONST(theta1);
    LOG_CONST(theta2);

    LOG_CONST(sin_phi);
    LOG_CONST(cos_phi);

    LOG_CONST(a);
    LOG_CONST(b);
    LOG_CONST(f * cos_phi);
    LOG_CONST(b * sin_phi);
    LOG_CONST(f_dash);
    LOG_CONST(A0);
    LOG_CONST(B0);
    LOG_CONST(B1);
    LOG_CONST(C0);
    LOG_CONST(C1);
    LOG_CONST(C2);
    LOG_CONST(x_dash);
    LOG_CONST(theta_dash);

    LOG_CONST(x_r);
    LOG_CONST(pos_s);
    LOG_CONST(x_s);
    LOG_CONST(z_s);
    LOG_CONST(d);

    std::vector<Eigen::Vector3d> mirror;

    std::ofstream file("ring_focus.txt");

    for (std::size_t i = 0; i < 100; ++i) {
        Angle beta = 2.0 * M_PI * i / 100.0 * 1.0_rad;

        Angle alpha_min = (M_PI_2 - std::atan2(z(-f_dash - ML) - d, -f_dash - ML - x_r)) * 1.0_rad;
        Angle alpha_max = (M_PI_2 - std::atan2(z(-f_dash) - d, -f_dash - x_r)) * 1.0_rad;
        Angle alpha_step = (alpha_max - alpha_min) / 99.0;
        for (std::size_t j = 0; j < 100; ++j) {
            Angle alpha = alpha_min + alpha_step * j;

            const double sin_alpha = std::sin(alpha);
            const double cos_alpha = std::cos(alpha);
            const double sin_beta = std::sin(beta);
            const double cos_beta = std::cos(beta);

            const Eigen::Matrix3d rotation
                = Eigen::AngleAxisd(-theta2, Eigen::Vector3d::UnitY()).toRotationMatrix();
            Eigen::Vector3d offset
                = rotation
                  * vectorize(x_r, d * cos_beta, d * sin_beta);
            //offset(2) = 0.0;

            Eigen::Vector3d pos_R
                = offset
                  + rotation
                        * (double(p(alpha, beta) / 1.0_m)
                            * vectorize(sin_alpha, cos_beta * cos_alpha, sin_beta * cos_alpha));

            file << pos_R(0) << ' ' << pos_R(1) << ' ' << pos_R(2) << std::endl;
        }
        file << std::endl;
    }

    return 0;
}
