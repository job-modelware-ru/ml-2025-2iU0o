#include "domain.h"
#include "wine_data_processor.h"
#include "svm.h"
#include <iostream>
#include <fstream>
using namespace std;

void RunPythonScript() {
    cout << "\nRunning Python script....\n";
    string command = "python.exe " + CONST::PYHON_SCRIPT_DIR;
    try {
        system(command.c_str());
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        exit(1);
    }
}

pair<vector<vector<double>>, vector<int>> PrepareData(const vector<Wine>& wines) {
    vector<vector<double>> X;
    vector<int> y;

    for (const auto& wine : wines) {
        vector<double> features = { wine.param.first, wine.param.second };
        X.push_back(features);
        y.push_back(wine.type == RED ? 1 : -1);
    }

    return { X, y };
}

void SaveWeights(const vector<double>& weights, double bias, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filename);

    for (auto& w : weights) file << w << ";";
    file << bias;
    file.close();
}

void SaveSupportVectors(const vector<vector<double>>& sv, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filename);
    
    bool is_first = true;
    for (auto& point : sv) {
        file << point[0] << ";" << point[1] << "\n";
    }
    file.close();
}

int main() {
    cout << "Starting SVM C++\n";
    cout << "Collecting data\n";
    WineDataProcessor train_processor;
    vector<Wine> wines = train_processor.GetWines();
    auto [X_train, y_train] = PrepareData(wines);

    double с = 1.0;
    double lr = 0.001;
    int epochs = 5000;
    cout << "Launching SVM\n";
    LinearSVM svm(с, lr, epochs);

    cout << "Trainig...\n";
    svm.Fit(X_train, y_train);

    cout << "Saving data\n";
    cout << "Weights: " << svm.GetWeights()[0] << "; " << svm.GetWeights()[1] << "\n";
    cout << "Bias: " << svm.GetBias() << "\n";
    SaveWeights(svm.GetWeights(), svm.GetBias(), "weights.csv");
       
    //RunPythonScript();
}