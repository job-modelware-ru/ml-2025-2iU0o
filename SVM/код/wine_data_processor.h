#pragma once

class WineDataProcessor {
public:
    WineDataProcessor(const pair<int, int> wine_data_range = { 0, CONST::DEFAULT_MAX_ROWS }) : wine_data_range_(wine_data_range) {
        vector<Wine> white_wines = ReadWineData(CONST::WHITE_WINE_RAW_DATA, WHITE);
        vector<Wine> red_wines = ReadWineData(CONST::RED_WINE_RAW_DATA, RED);

        wines_.insert(white_wines.begin(), white_wines.end());
        wines_.insert(red_wines.begin(), red_wines.end());

        SaveWinesToCSV();
    }

    vector<Wine> GetWines() {
        vector<Wine> wines;
        // Копируем элементы из set в vector
        for (const auto& wine : wines_)
            wines.push_back(wine);

        return wines;
    }
private:
    set<Wine> wines_;
    const pair<int, int> wine_data_range_;

    void SaveWinesToCSV() {
        ofstream file(CONST::SAVE_WINE_DATA);
        if (!file.is_open())
            throw runtime_error("Cannot open file: " + CONST::SAVE_WINE_DATA);

        file << "param_1;param_2;type\n";

        // Проходим по set и сохраняем данные
        for (const auto& wine : wines_)
            file << wine.param.first << ";" << wine.param.second << ";" << (wine.type == WineType::WHITE ? 1 : -1) << "\n";

        file.close();
    }

    vector<string> Split(const string& s) {
        vector<string> tokens;
        string token;
        istringstream token_stream(s);
        while (getline(token_stream, token, ';')) {
            if (!token.empty()) {
                if (token.front() == '"' && token.back() == '"') token = token.substr(1, token.size() - 2);
                tokens.push_back(move(token));
            }
        }
        return tokens;
    }

    vector<Wine> ReadWineData(const string& filename, WineType wine_type) {
        vector<Wine> wines;
        ifstream file(filename);

        if (!file.is_open()) {
            throw runtime_error("Cannot open file: " + filename);
        }

        string line;
        getline(file, line);
        wines.reserve(wine_data_range_.second - wine_data_range_.first);

        for (size_t rows_processed = wine_data_range_.first; rows_processed < wine_data_range_.second && getline(file, line); ++rows_processed) {
            if (line.empty()) continue;

            vector<string> tokens = Split(line);
            if (tokens.size() <= CONST::FEATURE_COUNT) continue;

            try {
                Wine wine;
                wine.param = { stod(tokens[1]), stod(tokens[2]) };
                //wine.param = { stod(tokens[1]), stod(tokens[6]) };
                wine.type = wine_type;
                wines.push_back(wine);
            }
            catch (const exception& e) {
                cerr << "Warning: Invalid data in row " << rows_processed << ": " << e.what() << endl;
            }
        }

        return wines;
    }
};

