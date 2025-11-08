#pragma once

class LinearSVM {

public:
    LinearSVM(double C = 1.0, double lr = 0.001, int epochs_ = 1000) : с_(C), learning_rate_(lr), epochs_(epochs_), bias_(0) {}

    void Fit(const vector<vector<double>>& X, const vector<int>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();

        mean_.assign(n_features, 0.0);
        std_.assign(n_features, 0.0);
        for (int j = 0; j < n_features; ++j) {
            for (int i = 0; i < n_samples; ++i)
                mean_[j] += X[i][j];
            mean_[j] /= n_samples;

            double var = 0.0;
            for (int i = 0; i < n_samples; ++i)
                var += pow(X[i][j] - mean_[j], 2);
            std_[j] = sqrt(var / n_samples);
        }

        vector<vector<double>> Xs(n_samples, vector<double>(n_features));
        for (int i = 0; i < n_samples; ++i)
            Xs[i] = Standardize(X[i]);

        weights_.assign(n_features, 0.0);
        bias_ = 0.0;

        vector<int> indices(n_samples);
        iota(indices.begin(), indices.end(), 0);
        mt19937 rng(42);

        double C_scaled = с_ * n_samples;

        for (int epoch = 0; epoch < epochs_; ++epoch) {
            shuffle(indices.begin(), indices.end(), rng);

            for (int idx : indices) {
                double decision = bias_;
                for (int j = 0; j < n_features; ++j)
                    decision += weights_[j] * Xs[idx][j];
                
                if (y[idx] * decision < 1) {
                    for (int j = 0; j < n_features; ++j)
                        weights_[j] += learning_rate_ * (C_scaled * y[idx] * Xs[idx][j] - weights_[j]);
                    bias_ += learning_rate_ * C_scaled * y[idx];
                }
                else {
                    for (int j = 0; j < n_features; ++j)
                        weights_[j] -= learning_rate_ * weights_[j];
                }
            }

            learning_rate_ *= 1.0 / (1.0 + 0.0001 * epoch);
        }

        for (auto& w : weights_) w = -w;
        bias_ = -bias_;
    }

    int Predict(const vector<double>& x) const {
        double decision = bias_;
        for (size_t j = 0; j < weights_.size(); ++j)
            decision += weights_[j] * x[j];
        return decision >= 0 ? 1 : -1;
    }

    vector<vector<double>> FindSupportVectors(const vector<vector<double>>& X, const vector<int>& y) {
        vector<vector<double>> support_vectors;
        for (int i = 0; i < X.size(); ++i) {
            vector<double> xs = Standardize(X[i]);
            double margin = y[i] * (bias_ + inner_product(weights_.begin(), weights_.end(), xs.begin(), 0.0));
            if (fabs(margin + 1.0) < CONST::M_EPS) {
                support_vectors.push_back(xs);
            }
        }
        return support_vectors;
    }
    
    vector<double> GetWeights() const { return weights_; }
    double GetBias() const { return bias_; }

private:
    vector<double> weights_;
    double bias_;
    double с_;
    double learning_rate_;
    int epochs_;

    vector<double> mean_;
    vector<double> std_;

    vector<double> Standardize(const vector<double>& x) const {
        vector<double> xs(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            xs[i] = (x[i] - mean_[i]) / (std_[i] > 1e-12 ? std_[i] : 1.0);
        return xs;
    }

    double Margin(const vector<double>& x, int true_label) {
        double decision = bias_;
        for (int i = 0; i < weights_.size(); ++i) {
            decision += weights_[i] * x[i];
        }
        return true_label * decision;
    }
};
