#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <execution>
#include <fstream>
#include <iostream>

#include <fftw3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <grid/algorithm.hpp>
#include <grid/bundle.hpp>
#include <grid/core.hpp>
#include <grid/linear.hpp>
#include <unit/double.hpp>

#include <string>
#include <unit/double.hpp>
#include <vector>

#include "constants.hpp"

struct MicroArea {
    Eigen::Vector3d pos;
    Eigen::Vector3d normal;

    Area dS;

    MicroArea(Length x, Length y, Length z, Eigen::Vector3d normal, Area dS)
        : pos{x / 1.0_m, y / 1.0_m, z / 1.0_m}, normal(normal), dS(dS) {}
    MicroArea()
        : pos(Eigen::Vector3d::Zero()), normal(Eigen::Vector3d::Zero()), dS(0.0_m2) {}
};

Length ellipse_radius(const Length& z)
{
    return b * std::sqrt(1.0 - std::pow(z / a, 2));
}
auto ellipse_normal(const Length& z)
{
    constexpr double coef = b * b / (a * a);
    return Eigen::Vector3d{0.0, 1.0, coef * z / ellipse_radius(z)}.normalized();
}

int main()
{
    //Grid::parallelize();

    //double NA;
    //Length exit_length;

    //Grid::DynamicRange<Length> detector_range{-0.5 * detector_length, 0.5 * detector_length, detector_pixel_num};
    //Grid::GridVector<Complex<WaveAmplitude>, Length, 2> detector{detector_range, detector_range};

    //// z = f - 100nm, f, f + 100nm
    //for (auto& translate : {-100.0_nm, 0.0_nm, 100.0_nm}) {
    //Length knifeedge_z = f + translate;
    //Length knife_focus_length = detector_pixel_num * lambda * (CD - translate) / detector_length / 2;

    //Grid::DynamicRange<Length> focus_range{-0.5 * focus_length, 0.5 * focus_length, focus_pixel_num};
    //Grid::GridVector<Complex<WaveAmplitude>, Length, 2> focus{focus_range, focus_range};

    //focus.fill(0.0 * amp_unit);

    //auto focus_zip = Grid::zip(focus.lines(), focus);
    //const std::size_t size = installed_mirror.size();
    //std::atomic_ulong c = 0;

    //auto zip = Grid::zip(installed_mirror, mirror_wave, mirror_effective_dS);
    //std::for_each(std::execution::par, zip.begin(), zip.end(), [&c, &size, &focus_zip, &knifeedge_z](auto x) {
    //auto [area, wave, dS] = x;

    //if (c++ % 10000 == 0) {
    //std::cout << c << " / " << size << std::endl;
    //}

    //auto coef = wave * dS / 1.0i / lambda;
    //for (auto [x, y, U] : focus_zip) {
    //Eigen::Vector3d r = Eigen::Vector3d{x / 1.0_m, y / 1.0_m, knifeedge_z / 1.0_m} - area.pos;
    //auto r_squared_norm = r.squaredNorm() * 1.0_m2;
    //U += coef * r.dot(area.normal) * 1.0_m / r_squared_norm * std::exp(1.0i * k * std::sqrt(r_squared_norm));
    //}
    //});

    //for (auto& x : focus.line(0)) {
    //for (auto& y : focus.line(1)) {
    //if (x <= 0.0_nm) {
    //focus.at(x, y) = 0.0 * amp_unit;
    //}
    //}
    //}

    //{
    //std::ofstream file("data_foucaulttest_ellipse/obstructed_" + std::to_string(int(translate / 1.0_nm)) + ".txt");

    //for (auto& x : focus.line(0)) {
    //for (auto& y : focus.line(1)) {
    //file << x / 1.0_m << ' '
    //<< y / 1.0_m << ' '
    //<< std::arg(focus.at(x, y)) << ' '
    //<< std::norm(focus.at(x, y)).value << std::endl;
    //}
    //file << std::endl;
    //}
    //}

    //fftw_plan plan
    //= fftw_plan_dft_2d(focus_pixel_num, focus_pixel_num,
    //reinterpret_cast<fftw_complex*>(focus.data()), reinterpret_cast<fftw_complex*>(detector.data()), FFTW_BACKWARD, FFTW_ESTIMATE);

    //fftw_execute(plan);
    //Grid::fftshift(detector);

    //{
    //std::ofstream file("data_foucaulttest_ellipse/detected_" + std::to_string(int(translate / 1.0_nm)) + ".txt");
    //for (auto& x : detector.line(0)) {
    //for (auto& y : detector.line(1)) {
    //file << x / 1.0_m << ' '
    //<< y / 1.0_m << ' '
    //<< std::arg(detector.at(x, y)) << ' '
    //<< std::norm(detector.at(x, y)).value / (detector_pixel_num * detector_pixel_num) << std::endl;
    //}
    //file << std::endl;
    //}
    //}
    //fftw_destroy_plan(plan);
    //}

    return 0;
}
