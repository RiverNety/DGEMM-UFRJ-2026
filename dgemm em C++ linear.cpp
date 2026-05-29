#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

int main() {
    int n = 1000;
    
    // matriz linear
    vector<int> a(n * n, 1);
    vector<int> b(n * n, 1);
    vector<int> c(n * n, 0);

    vector<double> tempo;

    for(int i = 0; i<5; i++) {
        fill(c.begin(), c.end(), 0);

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
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