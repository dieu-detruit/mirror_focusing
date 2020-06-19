#include <iostream>

#include <grid/algorithm.hpp>
#include <grid/bundle.hpp>
#include <grid/core.hpp>

int main()
{
    //Grid::parallelize();

    //{
    //Grid::GridVector<double, double, 1> dist{{-1.0, 1.0, 5}};

    //for (auto [x, v] : Grid::zip(dist.line(0), dist)) {
    //v = 4 * x;
    //}

    //for (auto& v : dist) {
    //std::cout << v << ' ';
    //}
    //std::cout << std::endl;

    //fftshift(dist);


    //for (auto& v : dist) {
    //std::cout << v << ' ';
    //}
    //std::cout << std::endl;

    //std::cout << std::endl;
    //}

    /*
     *  1  2 | 3       9 | 7 8
     *  4  5 | 6       --+-----
     *  -----+---  ->  3 | 1 2  
     *  7  8 | 9       6 | 4 5
     *
     *  1 += 4
     *  2 += 4
     *  3 += 1
     *  4 += 4
     *
     *  4 = 3 * 1 + 1
     *  1 = 3 * 1 - 2
     *
     *  1 2 3 4 5 6 7 8 9  -> 9 7 8 | 3 1 2 | 6 4 5
     *
     *  1  2 |  3  4        11 12 |  9 10
     *  4  6 |  7  8        ------+------
     *  -----+------   ->    3  4 |  1  2
     *  9 10 | 11 12         7  8 |  4  6
     *
     *  非対称は違う枠
     *
     *
     *   1  2 |  3  4        11 12 |  9 10
     *   5  6 |  7  8        15 16 | 13 14
     *  ------+------   ->   ------+------
     *   9 10 | 11 12         3  4 |  1  2
     *  13 14 | 15 16         7  8 |  5  6
     *
     *  1 += 10
     *  2 += 10
     *  3 += 6
     *  4 += 6
     *
     *  10 = 4 * 2 + 2
     *   6 = 4 * 2 - 2
     *
     *  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 -> 11 12 9 10 | 15 16 13 14 | 3 4 1 2 | 7 8 5 6
     *
     *   1  2  3 |  4  5       19 20 | 16 17 18
     *   6  7  8 |  9 10       24 25 | 21 22 23
     *  11 12 13 | 14 15       ------+---------
     *  ---------+-------  ->   4  5 |  1  2  3
     *  16 17 18 | 19 20        9 10 |  6  7  8
     *  21 22 23 | 24 25       14 15 | 11 12 13
     * 
     *  1 += 12
     *  2 += 12
     *  3 += 12
     *  4 += 7
     *  5 += 7
     *  9 += 7
     *  
     *  12 = 5 * 2 + 2
     *  7 =  5 * 2 - 3
     *
     *  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 -> 19 20 16 17 18 | 24 25 21 22 23 | 4 5 1 2 3 | 9 10 6 7 8 | 14 15 11 12 13
     * 
     * 結局 lineでrotateしてline内でrotateする2重の構造になっているだけ
     *
     */


    // 2D
    {
        Grid::DynamicRange<double> range{0.0, 5.0, 5};
        Grid::GridVector<double, double, 2> dist{range, range};

        // std::size_t a = Grid::Impl::get_rank_v<Grid::GridVector<double, double, 2>>;
        //std::cout << "a: " << a << std::endl;
        //hoge<Grid::Impl::get_rank_v<Grid::GridVector<double, double, 2>>> _;

        int i = 1;
        for (auto& v : dist) {
            v = i;
            ++i;
        }

        for (auto& x : dist.line(0)) {
            for (auto& y : dist.line(1)) {
                std::cout << dist.at(x, y) << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        Grid::fftshift(dist);

        for (auto& x : dist.line(0)) {
            for (auto& y : dist.line(1)) {
                std::cout << dist.at(x, y) << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
