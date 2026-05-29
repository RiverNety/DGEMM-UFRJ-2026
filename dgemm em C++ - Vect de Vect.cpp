#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int main() {
    int n = 4000;
    
    vector<vector<int>> a(n, vector<int>(n, 1));
    vector<vector<int>> b(n, vector<int>(n, 1));
    vector<vector<int>> c(n, vector<int>(n, 0));

    vector<double> tempo;

    for(int i = 0; i<5; i++) {
        for(auto& row : c) {
            fill(row.begin(), row.end(), 0);
        }

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        tempo.push_back(duration.count());
    }

    cout << "tempos" << endl;
    for(auto t : tempo) {
        cout << t << " s" << endl;
    }

    cout << "feito";
    return 0;
}