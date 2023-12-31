#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <chrono>
#include <string>

using namespace std;

const int grid_size = 160;
const int monte_carlo_num = 100000;
const int num_of_threads = 4;
const double A_1 = -2.0, B_1 = 5.0, A_2 = -2.0, B_2 = 5.0;
const double h_1 = (B_1 - A_1) / grid_size, h_2 = (B_2 - A_2) / grid_size;
const double eps = h_1 > h_2 ? h_1 * h_1 : h_2 * h_2;
double delta = 0.000001;

struct Point {
    double x;
    double y;
};

struct Rectangle {
    Point up_left;
    Point down_right;
};

bool point_in_D(const Point& point) {
    if ((point.x < 0.0) || (point.x > 3.0)) return false;
    else if ((point.y < 0.0) || point.y > 3.0) return false;
    else {
        if (point.x <= 2.0) return true;
        else if (point.y <= -3.0 * point.x + 9.0) return true;
        else return false;
    }
}

double get_a(const vector<vector<Point>>& points, const int i, const int j, const double step, const double eps) {
    auto point_1 = Point{ points[i][j].x - step * 0.5, points[i][j].y - step * 0.5 };
    auto point_2 = Point{ points[i][j].x - step * 0.5, points[i][j].y + step * 0.5 };
    if (point_1.y > 3.0 || point_2.y < 0.0) return 1 / eps;
    std::random_device rd_y;
    std::mt19937 gen_y(rd_y());
    std::uniform_real_distribution<> y(point_1.y, point_2.y);
    int counter = 0;
    for (int i = 0; i < (monte_carlo_num / 100); i++) {
        double y_t = y(gen_y);
        if (point_in_D(Point{ point_1.x, y_t })) counter++;
    }
    double l = step * counter / (monte_carlo_num / 100);
    return (l / step) + ((1 - l / step) / eps);
}

double get_b(const vector<vector<Point>>& points, const int i, const int j, const double step, const double eps) {
    auto point_1 = Point{ points[i][j].x - step * 0.5, points[i][j].y - step * 0.5 };
    auto point_2 = Point{ points[i][j].x + step * 0.5, points[i][j].y - step * 0.5 };
    if (point_1.x > 3.0 || point_2.x < 0.0) return 1 / eps;
    std::random_device rd_x;
    std::mt19937 gen_x(rd_x());
    std::uniform_real_distribution<> x(point_1.x, point_2.x);
    int counter = 0;
    for (int i = 0; i < (monte_carlo_num / 100); i++) {
        double x_t = x(gen_x);
        if (point_in_D(Point{ x_t, point_1.y })) counter++;
    }
    double l = step * counter / (monte_carlo_num / 100);
    return (l / step) + ((1 - l / step) / eps);
}

double get_intersection_area(Rectangle rect, double step) {
    if (rect.up_left.x > 3.0 || rect.down_right.x < 0.0 || rect.up_left.y < 0.0 || rect.down_right.y > 3.0) return 0.0;
    std::random_device rd_x, rd_y;
    std::mt19937 gen_x(rd_x()), gen_y(rd_y());
    std::uniform_real_distribution<> x(rect.up_left.x, rect.down_right.x);
    std::uniform_real_distribution<> y(rect.down_right.y, rect.up_left.y);
    int counter = 0;
    for (int i = 0; i < monte_carlo_num; i++) {
        double x_t = x(gen_x);
        double y_t = y(gen_y);
        if (point_in_D(Point{ x_t, y_t })) counter++;
    }
    return step * step * counter / monte_carlo_num;
}

void print_grid(const vector<vector<Point>>& points) {
    for (int i = 0; i < points.size(); i++) {
        for (int j = 0; j < points[i].size(); j++)
            cout << "(" << points[i][j].x << "," << points[i][j].y << ") ";
        cout << endl;
    }
}

double get_scalar(const vector<double>& vec_1, const vector<double>& vec_2, const double h_1, const double h_2) {
    double res = 0;
#pragma omp parallel for reduction(+:res)
    for (int i = 0; i < vec_1.size(); i++) {
        res += vec_1[i] * vec_2[i];
    }
    return res * h_1 * h_2;
}

double get_norm(const vector<double>& vec, const double h_1, const double h_2) {
    return sqrt(get_scalar(vec, vec, h_1, h_2));
}

void mult(const vector<vector<double>>& matrix, const vector<double>& w, vector<double>& result, const int grid_size) {
#pragma omp parallel for
    for (int i = grid_size + 2; i < matrix.size() - grid_size - 2; i++) {
        double tmp = 0.0;
        tmp += matrix[i][i] * w[i];
        tmp += matrix[i][i - 1] * w[i - 1];
        tmp += matrix[i][i + 1] * w[i + 1];
        tmp += matrix[i][i - grid_size - 1] * w[i - grid_size - 1];
        tmp += matrix[i][i + grid_size + 1] * w[i + grid_size + 1];
        result[i] = tmp;
    }
    return;
}

void sub(const vector<double>& vec_1, const vector<double>& vec_2, vector<double>& result) {
    #pragma omp parallel for
    for (int i = 0; i < vec_1.size(); i++) {
        result[i] = vec_1[i] - vec_2[i];
    }
    return;
}

void multiply(vector<double>& vec, const double tau) {
#pragma omp parallel for
    for (int i = 0; i < vec.size(); i++) {
        vec[i] *= tau;
    }
    return;
}

vector<double> resolve(const vector<vector<double>>& A, const vector<double>& F, const double delta, const double h_1, const double h_2, const int grid_size) {
    int counter = 0;

    vector<double> w_k(F.size(), 0.0);
    vector<double> w_k_1(F.size());
    vector<double> r_k(w_k_1.size());
    vector<double> A_r_k(r_k.size());
    vector<double> w_k_1_w_k(w_k_1.size());

    double A_r_k_norm, tau, d;

    do {
        mult(A, w_k, r_k, grid_size);
        sub(r_k, F, r_k);
        double disc = get_norm(r_k, h_1, h_2);
        mult(A, r_k, A_r_k, grid_size);
        A_r_k_norm = get_norm(A_r_k, h_1, h_2);
        tau = get_scalar(A_r_k, r_k, h_1, h_2) / (A_r_k_norm * A_r_k_norm);
        multiply(r_k, tau);
        sub(w_k, r_k, w_k_1);
        sub(w_k_1, w_k, w_k_1_w_k);
        d = get_norm(w_k_1_w_k, h_1, h_2);
        if (counter % 500 == 0) {
            cout << d << endl;
        }
        if (d < delta) {
            cout << counter << endl;
            break;
        }
        else w_k = w_k_1;
        counter++;
    } while (true);
    return w_k_1;
}

int main()
{
    cout << "start" << endl;
    auto start = chrono::steady_clock::now();
    omp_set_num_threads(num_of_threads);
    vector<double> x_vals_in(grid_size + 1);
    vector<double> y_vals_in(grid_size + 1);
    for (int i = 0; i < grid_size + 1; i++) {
        x_vals_in[i] = A_1 + i * h_1;
        y_vals_in[i] = A_2 + i * h_2;
    }
    vector<vector<Point>> points(grid_size + 1, vector<Point>(grid_size + 1, Point{ 0.0, 0.0 }));
    for (int i = 0; i < grid_size + 1; i++) {
        for (int j = 0; j < grid_size + 1; j++) {
            points[i][j] = Point{ x_vals_in[j], y_vals_in[i] };
        }
    }

    vector<vector<double>> A((grid_size + 1) * (grid_size + 1), vector<double>((grid_size + 1) * (grid_size + 1), 0.0));

    vector<double> F((grid_size + 1) * (grid_size + 1), 0.0);
#pragma omp parallel for
    for (int i = 1; i < grid_size; i++) {
        for (int j = 1; j < grid_size; j++) {
            Point up_left = Point{ points[i][j].x - h_1 / 2, points[i][j].y + h_2 / 2 };
            Point down_right = Point{ points[i][j].x + h_1 / 2, points[i][j].y - h_2 / 2 };
            auto rect = Rectangle{ up_left, down_right };
            F[i * (grid_size + 1) + j] = get_intersection_area(rect, h_1) / (h_1 * h_2);
        }
    }
#pragma omp parallel for
    for (int i = 1; i < grid_size; i++) {
        for (int j = 1; j < grid_size; j++) {
            double a_i1_j = get_a(points, i + 1, j, h_2, eps);
            double a_ij = get_a(points, i, j, h_2, eps);
            double b_i_j1 = get_b(points, i, j + 1, h_1, eps);
            double b_ij = get_b(points, i, j, h_1, eps);
            A[i * (grid_size + 1) + j][i * (grid_size + 1) + j] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
            if (i == 1) {
                A[i * (grid_size + 1) + j][(i + 1) * (grid_size + 1) + j] = -a_i1_j / (h_1 * h_1);
                if (j == 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
                else if (j == grid_size - 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                }
                else {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
            }
            else if (i == grid_size - 1) {
                A[i * (grid_size + 1) + j][(i - 1) * (grid_size + 1) + j] = -a_ij / (h_1 * h_1);;
                if (j == 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
                else if (j == grid_size - 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                }
                else {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
            }
            else {
                A[i * (grid_size + 1) + j][(i - 1) * (grid_size + 1) + j] = -a_ij / (h_1 * h_1);;
                A[i * (grid_size + 1) + j][(i + 1) * (grid_size + 1) + j] = -a_i1_j / (h_1 * h_1);
                if (j == 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
                else if (j == grid_size - 1) {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                }
                else {
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j - 1] = -b_ij / (h_2 * h_2);
                    A[i * (grid_size + 1) + j][i * (grid_size + 1) + j + 1] = -b_i_j1 / (h_2 * h_2);
                }
            }
        }
    }
    cout << endl;
    delta = 0.000001;
    auto result = resolve(A, F, delta, h_1, h_2, grid_size);
    auto diff = chrono::steady_clock::now() - start;
    cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
    ofstream output("result_4.txt");
    for (int i = 0; i < grid_size + 1; i++) {
        for (int j = 0; j < grid_size + 1; j++) {
            output << result[i * (grid_size + 1) + j] << " ";
        }
        output << endl;
    }
    return 0;
}
