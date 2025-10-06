#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

enum WineType { WHITE, RED };

struct Wine {
    pair<double, double> param;
    WineType type;
};

namespace CONST {
    const size_t DEFAULT_MAX_ROWS = 2;
    const pair<size_t, size_t> PREDICTION_RANGE = { DEFAULT_MAX_ROWS, DEFAULT_MAX_ROWS * 2 };
    const size_t FEATURE_COUNT = 3;
    const double M_EPS = 1e-6;

    const string WHITE_WINE_RAW_DATA = "wine_quality_white.csv";
    const string RED_WINE_RAW_DATA = "wine_quality_red.csv";
    const string SAVE_WINE_DATA = "wine_data.csv";
    const string SAVE_SUPPORT_VECTORS_DATA = "support_vector_data.csv";

    const string PYHON_SCRIPT_DIR = "C:\\Users\\user\\source\\repos\\py-support-vector-machine\\main.py";
}

bool operator<(const Wine& lhs, const Wine& rhs) {
    if (lhs.param.first != rhs.param.first)
        return lhs.param.first < rhs.param.first;
    if (lhs.param.second != rhs.param.second)
        return lhs.param.second < rhs.param.second;

    return lhs.type < rhs.type;
}