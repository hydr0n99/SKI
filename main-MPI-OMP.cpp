#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include <string>
#include <mpi.h>
#include <omp.h>

using namespace std;

int nproc, nrank;
const int monte_carlo_num = 100000;
const double delta = 0.000001;
const int num_of_threads = 4;

struct Point {
    double x;
    double y;

    double* from_point() const {
        auto coords = new double[2];
        coords[0] = x;
        coords[1] = y;
        return coords;
    }
    void to_point(double* coords) {
        x = coords[0];
        y = coords[1];
        return;
    }
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

void mult(const vector<double*>& A, const vector<double>& w, vector<double>& result, const int grid_size) {
    #pragma omp parallel for
    for (int i = 0; i < A.size(); i++) {
        int idx = round(A[i][0]);
        if ((idx < grid_size + 2) || (idx > (grid_size + 1) * grid_size - 2)) continue;
        double tmp = 0.0;
        tmp += w[idx] * A[i][3];
        tmp += w[idx - 1] * A[i][2];
        tmp += w[idx + 1] * A[i][4];
        tmp += w[idx - grid_size - 1] * A[i][1];
        tmp += w[idx + grid_size + 1] * A[i][5];
        result[i] = tmp;
    }

}

void sub(const vector<double>& vec_1, const vector<pair<int, double>>& vec_2, vector<double>& result) {
    #pragma omp parallel for
    for (int i = 0; i < vec_1.size(); i++) {
        result[i] = vec_1[i] - vec_2[i].second;
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

void build_vector_4(const int row_count, const int col_count, vector<double>& result, const vector<double>& vec_0, const vector<double>& vec_1, const vector<double>& vec_2, const vector<double>& vec_3) {
    #pragma omp parallel for
    for (int i = 0; i < row_count; i++) {
        for (int j = 0; j < col_count; j++) {
            result[i * (col_count * 2 - 1) + j] = vec_0[i * col_count + j];
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < row_count; i++) {
        for (int j = 1; j < col_count; j++) {
            result[i * (2 * col_count - 1) + col_count + j - 1] = vec_1[i * (col_count - 1) + j - 1];
        }
    }
    #pragma omp parallel for
    for (int i = 1; i < row_count; i++) {
        for (int j = 0; j < col_count; j++) {
            result[(i + row_count - 1) * (col_count * 2 - 1) + j] = vec_2[(i - 1) * col_count + j];
        }
    }
    #pragma omp parallel for
    for (int i = 1; i < row_count; i++) {
        for (int j = 1; j < col_count; j++) {
            result[(i + row_count - 1) * (col_count * 2 - 1) + col_count + j - 1] = vec_3[(i - 1) * (col_count - 1) + j - 1];
        }
    }
}

void build_vector_2(const int row_count, const int col_count, vector<double>& result, const vector<double>& vec_0, const vector<double>& vec_1) {
    #pragma omp parallel for
    for (int i = 0; i < vec_0.size(); i++)
        result[i] = vec_0[i];
    #pragma omp parallel for
    for (int i = 0; i < vec_1.size(); i++)
        result[vec_0.size() + i] = vec_1[i];
}

int main(int argc, char **argv)
{
    auto start = chrono::steady_clock::now();

    int grid_size = stoi(argv[1]);

    int row_count, col_count;
    double A_1 = -2.0, B_1 = 5.0, A_2 = -2.0, B_2 = 5.0;
    const double h_1 = (B_1 - A_1) / grid_size, h_2 = (B_2 - A_2) / grid_size;
    const double eps = h_1 > h_2 ? h_1 * h_1 : h_2 * h_2;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &nrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    omp_set_num_threads(num_of_threads);
    MPI_Status status;

    if (nproc == 2) {
        row_count = grid_size / 2 + 1;
        col_count = grid_size + 1;

        vector<double> x_vals_in(col_count);
        vector<double> y_vals_in;

        for (int j = 0; j < col_count; j++)
            x_vals_in[j] = A_1 + j * h_1;
        if (nrank == 0) {
            y_vals_in = vector<double>(row_count + 1);
            #pragma omp parallel for
            for (int i = 0; i < row_count + 1; i++)
                y_vals_in[i] = A_2 + i * h_2;
        }
        else {
            y_vals_in = vector<double>(row_count);
            #pragma omp parallel for
            for (int i = 0; i < row_count; i++)
                y_vals_in[i] = A_2 + (B_2 - A_2) / 2 + i * h_2;
        }

        vector<vector<Point>> points(y_vals_in.size(), vector<Point>(x_vals_in.size(), Point{ 0.0, 0.0 }));
        #pragma omp parallel for
        for (int i = 0; i < y_vals_in.size(); i++) {
            for (int j = 0; j < x_vals_in.size(); j++) {
                points[i][j] = Point{ x_vals_in[j], y_vals_in[i] };
            }
        }
        
        vector<double*> A;
        if (nrank == 0) {
            A = vector<double*>(row_count * col_count);
            #pragma omp parallel for
            for (int i = 0; i < row_count; i++) {
                for (int j = 0; j < col_count; j++) {
                    A[i * col_count + j] = new double[6];
                    A[i * col_count + j][0] = i * col_count + j;
                    if (i == 0 || j == 0 || j == (col_count - 1)) {
                        for (int k = 1; k < 6; k++)
                            A[i * col_count + j][k] = 0.0;
                    }
                    else {
                        double a_i1_j = get_a(points, i + 1, j, h_2, eps);
                        double a_ij = get_a(points, i, j, h_2, eps);
                        double b_i_j1 = get_b(points, i, j + 1, h_1, eps);
                        double b_ij = get_b(points, i, j, h_1, eps);

                        A[i * col_count + j][3] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
                        A[i * col_count + j][5] = -a_i1_j / (h_1 * h_1);

                        if (i != 1)
                            A[i * col_count + j][1] = -a_ij / (h_1 * h_1);
                        else
                            A[i * col_count + j][1] = 0.0;

                        if (j != 1)
                            A[i * col_count + j][2] = -b_ij / (h_2 * h_2);
                        else
                            A[i * col_count + j][2] = 0.0;

                        if (j != col_count - 1)
                            A[i * col_count + j][4] = -b_i_j1 / (h_2 * h_2);
                        else
                            A[i * col_count + j][4] = 0.0;
                    }
                }
            }
        }
        else {
            A = vector<double*>((row_count - 1) * col_count);
            #pragma omp parallel for
            for (int i = 1; i < row_count; i++) {
                for (int j = 0; j < col_count; j++) {
                    A[(i - 1) * col_count + j] = new double[6];
                    A[(i - 1) * col_count + j][0] = (i + row_count - 1) * col_count + j;
                    if (j == 0 || i == (row_count - 1) || j == (col_count - 1)) {
                        for (int k = 1; k < 6; k++)
                            A[(i - 1) * col_count + j][k] = 0.0;
                    }
                    else {
                        double a_i1_j = get_a(points, i + 1, j, h_2, eps);
                        double a_ij = get_a(points, i, j, h_2, eps);
                        double b_i_j1 = get_b(points, i, j + 1, h_1, eps);
                        double b_ij = get_b(points, i, j, h_1, eps);
                        A[(i - 1) * col_count + j][3] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
                        A[(i - 1) * col_count + j][1] = -a_ij / (h_1 * h_1);
                        A[(i - 1) * col_count + j][4] = -b_i_j1 / (h_2 * h_2);

                        if (j != 1)
                            A[(i - 1) * col_count + j][2] = -b_ij / (h_2 * h_2);
                        else
                            A[(i - 1) * col_count + j][2] = 0.0;

                        if (i != row_count - 2)
                            A[(i - 1) * col_count + j][5] = -a_i1_j / (h_1 * h_1);
                        else
                            A[(i - 1) * col_count + j][5] = 0.0;

                        if (j != col_count - 1)
                            A[(i - 1) * col_count + j][4] = -b_i_j1 / (h_2 * h_2);
                        else
                            A[(i - 1) * col_count + j][4] = 0.0;
                    }
                }
            }
        }
        vector<pair<int, double>> F;
        if (nrank == 0) {
            F = vector<pair<int, double>>(row_count * col_count, make_pair(0, 0.0));
            #pragma omp parallel for
            for (int i = 0; i < row_count; i++) {
                for (int j = 0; j < col_count; j++) {
                    if (!(i == 0 || j == 0 || j == (col_count - 1))) {
                        Point up_left = Point{ points[i][j].x - h_1 / 2, points[i][j].y + h_2 / 2 };
                        Point down_right = Point{ points[i][j].x + h_1 / 2, points[i][j].y - h_2 / 2 };
                        auto rect = Rectangle{ up_left, down_right };
                        auto sqr = get_intersection_area(rect, h_1) / (h_1 * h_2);
                        F[i * col_count + j].second = sqr;
                    }
                    F[i * col_count + j].first = i * col_count + j;
                }
            }
            
        }
        else {
            F = vector<pair<int, double>>((row_count - 1) * col_count, make_pair(0, 0.0));
            #pragma omp parallel for
            for (int i = 1; i < row_count; i++) {
                for (int j = 0; j < col_count; j++) {
                    if (!(i == row_count - 1 || j == 0 || j == (col_count - 1))) {
                        Point up_left = Point{ points[i][j].x - h_1 / 2, points[i][j].y + h_2 / 2 };
                        Point down_right = Point{ points[i][j].x + h_1 / 2, points[i][j].y - h_2 / 2 };
                        auto rect = Rectangle{ up_left, down_right };
                        auto sqr = get_intersection_area(rect, h_1) / (h_1 * h_2);
                        F[(i - 1) * col_count + j].second = sqr;
                    }
                    F[(i - 1) * col_count + j].first = (i + row_count - 1) * col_count + j;
                }
            }
        }

        vector<double> result((grid_size + 1)* (grid_size + 1));
        int counter = 0;
        int full_len = (grid_size + 1) * (grid_size + 1);
        vector<double> w_k(full_len, 0.0);
        vector<double> w_k_1(w_k.size(), 0.0);
        vector<double> r_k(w_k.size());
        vector<double> r_k_this(F.size());
        vector<double> A_r_k(w_k.size());
        vector<double> A_r_k_this(F.size());
        vector<double> w_k_1_w_k(w_k.size());

        vector<int> sizes(2, 0);
        sizes[0] = row_count * col_count;
        sizes[1] = (row_count - 1) * col_count;
        double A_r_k_norm, tau, d, scalar;

        do {
            mult(A, w_k, r_k_this, grid_size);
            
            sub(r_k_this, F, r_k_this);

            if (nrank == 0) {
                vector<double> r_k_1(sizes[1], 0.0);
                MPI_Recv(r_k_1.data(), sizes[1], MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);

                build_vector_2(row_count, col_count, r_k, r_k_this, r_k_1);

                MPI_Send(r_k.data(), r_k.size(), MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);
            }
            else {
                MPI_Send(r_k_this.data(), r_k_this.size(), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                MPI_Recv(r_k.data(), r_k.size(), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            mult(A, r_k, A_r_k_this, grid_size);
            if (nrank == 0) {
                vector<double> A_r_k_1(sizes[1], 0.0);
                MPI_Recv(A_r_k_1.data(), sizes[1], MPI_DOUBLE, 1, 5, MPI_COMM_WORLD, &status);

                build_vector_2(row_count, col_count, A_r_k, A_r_k_this, A_r_k_1);
                MPI_Send(A_r_k.data(), A_r_k.size(), MPI_DOUBLE, 1, 6, MPI_COMM_WORLD);
            }
            else {
                MPI_Send(A_r_k_this.data(), A_r_k_this.size(), MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
                MPI_Recv(A_r_k.data(), A_r_k.size(), MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
            }
            if (nrank == 0) {
                A_r_k_norm = get_norm(A_r_k, h_1, h_2);
                MPI_Recv(&scalar, 1, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, &status);
            }
            else {
                scalar = get_scalar(A_r_k, r_k, h_1, h_2);
                MPI_Send(&scalar, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (nrank == 0) {
                tau = scalar / (A_r_k_norm * A_r_k_norm);
                multiply(r_k, tau);
                sub(w_k, r_k, w_k_1);
                sub(w_k_1, w_k, w_k_1_w_k);
                d = get_norm(w_k_1_w_k, h_1, h_2);
                if (counter % 500 == 0) {
                    cout << d << endl;
                }
                if (d < delta) {
                    uint8_t flag = 1;
                    MPI_Send(&flag, 1, MPI_UINT8_T, 1, 7, MPI_COMM_WORLD);
                    cout << counter << endl;
                    for (int i = 0; i < w_k_1.size(); i++)
                        result[i] = w_k_1[i];
                    break;
                }
                else {
                    uint8_t flag = 0;
                    MPI_Send(&flag, 1, MPI_UINT8_T, 1, 7, MPI_COMM_WORLD);

                    MPI_Send(w_k_1.data(), w_k_1.size(), MPI_DOUBLE, 1, 8, MPI_COMM_WORLD);
                    w_k = w_k_1;
                }
            }
            else {
                uint8_t flag;
                MPI_Recv(&flag, 1, MPI_UINT8_T, 0, 7, MPI_COMM_WORLD, &status);
                if (flag)
                    break;
                else
                    MPI_Recv(w_k.data(), w_k.size(), MPI_DOUBLE, 0, 8, MPI_COMM_WORLD, &status);
            }
            counter++;
        } while (true);
        if (nrank == 0) {
            auto diff = chrono::steady_clock::now() - start;
            cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
        }
        for (int i = 0; i < A.size(); i++) {
            delete A[i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        if (nrank == 0) {
            ofstream output("result_MPI.txt");
            for (int i = 0; i < grid_size + 1; i++) {
                for (int j = 0; j < grid_size + 1; j++) {
                    output << result[i * (grid_size + 1) + j] << " ";
                }
                output << endl;
            }
        }
    }
    else {
        row_count = grid_size / 2 + 1;
        col_count = grid_size / 2 + 1;

        vector<double> x_vals_in(col_count);
        vector<double> y_vals_in;

        if (nrank == 0) {
            x_vals_in = vector<double>(col_count + 1);
            y_vals_in = vector<double>(row_count + 1);
            #pragma omp parallel for
            for (int j = 0; j < col_count + 1; j++)
                x_vals_in[j] = A_1 + j * h_1;
            #pragma omp parallel for
            for (int i = 0; i < row_count + 1; i++)
                y_vals_in[i] = A_2 + i * h_2;
        }
        else if (nrank == 1) {
            x_vals_in = vector<double>(col_count);
            y_vals_in = vector<double>(row_count + 1);
            #pragma omp parallel for
            for (int j = 0; j < col_count; j++)
                x_vals_in[j] = A_1 + (B_1 - A_1) / 2 + j * h_1;
            #pragma omp parallel for
            for (int i = 0; i < row_count + 1; i++)
                y_vals_in[i] = A_2 + i * h_2;
        }
        else if (nrank == 2) {
            x_vals_in = vector<double>(col_count + 1);
            y_vals_in = vector<double>(row_count);
            #pragma omp parallel for
            for (int j = 0; j < col_count + 1; j++)
                x_vals_in[j] = A_1 + j * h_1;
            #pragma omp parallel for
            for (int i = 0; i < row_count; i++)
                y_vals_in[i] = A_2 + (B_2 - A_2) / 2 + i * h_2;
        }
        else {
            x_vals_in = vector<double>(col_count);
            y_vals_in = vector<double>(row_count);
            #pragma omp parallel for
            for (int j = 0; j < col_count; j++)
                x_vals_in[j] = A_1 + (B_1 - A_1) / 2 + j * h_1;
            #pragma omp parallel for
            for (int i = 0; i < row_count; i++)
                y_vals_in[i] = A_2 + (B_2 - A_2) / 2 + i * h_2;
        }

        vector<vector<Point>> points(y_vals_in.size(), vector<Point>(x_vals_in.size(), Point{ 0.0, 0.0 }));
        #pragma omp parallel for
        for (int i = 0; i < y_vals_in.size(); i++) {
            for (int j = 0; j < x_vals_in.size(); j++) {
                points[i][j] = Point{ x_vals_in[j], y_vals_in[i] };
            }
        }

        vector<double*> A;
        if (nrank == 0) {
            A = vector<double*>(row_count * col_count);
            #pragma omp parallel for
            for (int i = 0; i < row_count; i++) {
                for (int j = 0; j < col_count; j++) {
                    A[i * col_count + j] = new double[6];
                    if (i == 0 || j == 0) {
                        A[i * col_count + j][0] = i * (col_count * 2 - 1) + j;
                        for (int k = 1; k < 6; k++)
                            A[i * col_count + j][k] = 0.0;
                    }
                    else {
                        double a_i1_j = get_a(points, i + 1, j, h_2, eps);
                        double a_ij = get_a(points, i, j, h_2, eps);
                        double b_i_j1 = get_b(points, i, j + 1, h_1, eps);
                        double b_ij = get_b(points, i, j, h_1, eps);
                        A[i * col_count + j][0] = i * (col_count * 2 - 1) + j;
                        
                        
                        A[i * col_count + j][3] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
                        A[i * col_count + j][4] = -b_i_j1 / (h_2 * h_2);
                        A[i * col_count + j][5] = -a_i1_j / (h_1 * h_1);

                        if (i != 1)
                            A[i * col_count + j][1] = -a_ij / (h_1 * h_1);
                        else
                            A[i * col_count + j][1] = 0.0;

                        if (j != 1)
                            A[i * col_count + j][2] = -b_ij / (h_2 * h_2);
                        else
                            A[i * col_count + j][2] = 0.0;
                    }
                }
            }
        }
        else if (nrank == 1) {
            A = vector<double*>(row_count * (col_count - 1));
            #pragma omp parallel
            for (int i = 0; i < row_count; i++) {
                for (int j = 1; j < col_count; j++) {
                    A[i * (col_count - 1) + j - 1] = new double[6];
                    if (i == 0 || j == (col_count - 1)) {
                        A[i * (col_count - 1) + j - 1][0] = i * (2 * col_count - 1) + col_count + j - 1;
                        for (int k = 1; k < 6; k++)
                            A[i * (col_count - 1) + j - 1][k] = 0.0;
                    }
                    else {
                        double a_i1_j = get_a(points, i + 1, j, h_2, eps);
                        double a_ij = get_a(points, i, j, h_2, eps);
                        double b_i_j1 = get_b(points, i, j + 1, h_1, eps);
                        double b_ij = get_b(points, i, j, h_1, eps);
                        A[i * (col_count - 1) + j - 1][0] = i * (2 * col_count - 1) + col_count + j - 1;
                        A[i * (col_count - 1) + j - 1][3] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
                        A[i * (col_count - 1) + j - 1][2] = -b_ij / (h_2 * h_2);
                        A[i * (col_count - 1) + j - 1][5] = -a_i1_j / (h_1 * h_1);

                        if (i != 1)
                            A[i * (col_count - 1) + j - 1][1] = -a_ij / (h_1 * h_1);
                        else
                            A[i * (col_count - 1) + j - 1][1] = 0.0;

                        if (j != col_count - 2)
                            A[i * (col_count - 1) + j - 1][4] = -b_i_j1 / (h_2 * h_2);
                        else
                            A[i * (col_count - 1) + j - 1][4] = 0.0;
                    }
                }
            }
        }
        else if (nrank == 2) {
            A = vector<double*>((row_count - 1) * col_count);
            #pragma omp parallel for
            for (int i = 1; i < row_count; i++) {
                for (int j = 0; j < col_count; j++) {
                    A[(i - 1) * col_count + j] = new double[6];
                    if (j == 0 || i == (row_count - 1)) {
                        A[(i - 1) * col_count + j][0] = (i + row_count - 1) * (col_count * 2 - 1) + j;
                        for (int k = 1; k < 6; k++)
                            A[(i - 1) * col_count + j][k] = 0.0;
                    }
                    else {
                        double a_i1_j = get_a(points, i + 1, j, h_2, eps);
                        double a_ij = get_a(points, i, j, h_2, eps);
                        double b_i_j1 = get_b(points, i, j + 1, h_1, eps);
                        double b_ij = get_b(points, i, j, h_1, eps);
                        A[(i - 1) * col_count + j][0] = (i + row_count - 1) * (col_count * 2 - 1) + j;
                        A[(i - 1) * col_count + j][3] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
                        A[(i - 1) * col_count + j][1] = -a_ij / (h_1 * h_1);
                        A[(i - 1) * col_count + j][4] = -b_i_j1 / (h_2 * h_2);

                        if (j != 1)
                            A[(i - 1) * col_count + j][2] = -b_ij / (h_2 * h_2);
                        else
                            A[(i - 1) * col_count + j][2] = 0.0;

                        if (i != row_count - 2)
                            A[(i - 1) * col_count + j][5] = -a_i1_j / (h_1 * h_1);
                        else {
                            A[(i - 1) * col_count + j][5] = 0.0;
                        }

                    }
                }
            }
        }
        else {
            A = vector<double*>((row_count - 1) * (col_count - 1));
            #pragma omp parallel for
            for (int i = 1; i < row_count; i++) {
                for (int j = 1; j < col_count; j++) {
                    A[(i - 1) * (col_count - 1) + j - 1] = new double[6];
                    if (i == (row_count - 1) || j == (col_count - 1)) {
                        A[(i - 1) * (col_count - 1) + j - 1][0] = (i + row_count - 1) * (col_count * 2 - 1) + col_count + j - 1;
                        for (int k = 1; k < 6; k++)
                            A[(i - 1) * (col_count - 1) + j - 1][k] = 0.0;
                    }
                    else {
                        double a_i1_j = get_a(points, i + 1, j, h_2, eps);
                        double a_ij = get_a(points, i, j, h_2, eps);
                        double b_i_j1 = get_b(points, i, j + 1, h_1, eps);
                        double b_ij = get_b(points, i, j, h_1, eps);
                        A[(i - 1) * (col_count - 1) + j - 1][0] = (i + row_count - 1) * (col_count * 2 - 1) + col_count + j - 1;
                        A[(i - 1) * (col_count - 1) + j - 1][3] = (a_i1_j + a_ij) / (h_1 * h_1) + (b_i_j1 + b_ij) / (h_2 * h_2);
                        A[(i - 1) * (col_count - 1) + j - 1][2] = -b_ij / (h_2 * h_2);
                        A[(i - 1) * (col_count - 1) + j - 1][1] = -a_ij / (h_1 * h_1);

                        if (j != col_count - 2)
                            A[(i - 1) * (col_count - 1) + j - 1][4] = -b_i_j1 / (h_2 * h_2);
                        else
                            A[(i - 1) * (col_count - 1) + j - 1][4] = 0.0;

                        if (i != row_count - 2)
                            A[(i - 1) * (col_count - 1) + j - 1][5] = -a_i1_j / (h_1 * h_1);
                        else {
                            A[(i - 1) * (col_count - 1) + j - 1][5] = 0.0;
                        }
                    }
                }
            }
        }
        vector<pair<int, double>> F;
        if (nrank == 0) {
            F = vector<pair<int, double>>(row_count * col_count, make_pair(0, 0.0));
            #pragma omp parallel for
            for (int i = 0; i < row_count; i++) {
                for (int j = 0; j < col_count; j++) {
                    if (!(i == 0 || j == 0)) {
                        Point up_left = Point{ points[i][j].x - h_1 / 2, points[i][j].y + h_2 / 2 };
                        Point down_right = Point{ points[i][j].x + h_1 / 2, points[i][j].y - h_2 / 2 };
                        auto rect = Rectangle{ up_left, down_right };
                        auto sqr = get_intersection_area(rect, h_1) / (h_1 * h_2);
                        F[i * col_count + j].second = sqr;
                    }
                    F[i * col_count + j].first = i * (col_count * 2 - 1) + j;
                }
            }
        }
        else if (nrank == 1) {
            F = vector<pair<int, double>>(row_count * (col_count - 1), make_pair(0, 0.0));
            #pragma omp parallel for
            for (int i = 0; i < row_count; i++) {
                for (int j = 1; j < col_count; j++) {
                    if (!(i == 0 || j == col_count - 1)) {
                        Point up_left = Point{ points[i][j].x - h_1 / 2, points[i][j].y + h_2 / 2 };
                        Point down_right = Point{ points[i][j].x + h_1 / 2, points[i][j].y - h_2 / 2 };
                        auto rect = Rectangle{ up_left, down_right };
                        auto sqr = get_intersection_area(rect, h_1) / (h_1 * h_2);
                        F[i * (col_count - 1) + j - 1].second = sqr;
                    }
                    F[i * (col_count - 1) + j - 1].first = i * (2 * col_count - 1) + col_count + j - 1;
                }
            }
        }
        else if (nrank == 2) {
            F = vector<pair<int, double>>((row_count - 1) * col_count, make_pair(0, 0.0));
            #pragma omp parallel for
            for (int i = 1; i < row_count; i++) {
                for (int j = 0; j < col_count; j++) {
                    if (!(i == row_count - 1 || j == 0)) {
                        Point up_left = Point{ points[i][j].x - h_1 / 2, points[i][j].y + h_2 / 2 };
                        Point down_right = Point{ points[i][j].x + h_1 / 2, points[i][j].y - h_2 / 2 };
                        auto rect = Rectangle{ up_left, down_right };
                        auto sqr = get_intersection_area(rect, h_1) / (h_1 * h_2);
                        F[(i - 1) * col_count + j].second = sqr;
                    }
                    F[(i - 1) * col_count + j].first = (i + row_count - 1) * (col_count * 2 - 1) + j;
                }
            }
        }
        else {
            F = vector<pair<int, double>>((row_count - 1) * (col_count - 1), make_pair(0, 0.0));
            #pragma omp parallel for
            for (int i = 1; i < row_count; i++) {
                for (int j = 1; j < col_count; j++) {
                    if (!(i == row_count - 1 || j == row_count)) {
                        Point up_left = Point{ points[i][j].x - h_1 / 2, points[i][j].y + h_2 / 2 };
                        Point down_right = Point{ points[i][j].x + h_1 / 2, points[i][j].y - h_2 / 2 };
                        auto rect = Rectangle{ up_left, down_right };
                        auto sqr = get_intersection_area(rect, h_1) / (h_1 * h_2);
                        F[(i - 1) * (col_count - 1) + j - 1].second = sqr;
                    }
                    F[(i - 1) * (col_count - 1) + j - 1].first = (i + row_count - 1) * (col_count * 2 - 1) + col_count + j - 1;
                }
            }
        }
        
        vector<double> result((grid_size + 1) * (grid_size + 1));
        int counter = 0;
        int full_len = (grid_size + 1) * (grid_size + 1);
        vector<double> w_k(full_len, 0.0);
        vector<double> w_k_1(w_k.size(), 0.0);
        vector<double> r_k(w_k.size());
        vector<double> r_k_this(F.size());
        vector<double> A_r_k(w_k.size());
        vector<double> A_r_k_this(F.size());
        vector<double> w_k_1_w_k(w_k.size());
        
        vector<int> sizes(4, 0);
        sizes[0] = row_count * col_count;
        sizes[1] = row_count * (col_count - 1);
        sizes[2] = (row_count - 1) * col_count;
        sizes[3] = (row_count - 1) * (col_count - 1);
        double A_r_k_norm, tau, d, scalar;
        do {
            
            mult(A, w_k, r_k_this, grid_size);
            sub(r_k_this, F, r_k_this);
            if (nrank == 0) {
                vector<double> r_k_1(sizes[1], 0.0), r_k_2(sizes[2], 0.0), r_k_3(sizes[3], 0.0);
                MPI_Recv(r_k_1.data(), sizes[1], MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(r_k_2.data(), sizes[2], MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(r_k_3.data(), sizes[3], MPI_DOUBLE, 3, 2, MPI_COMM_WORLD, &status);
                
                build_vector_4(row_count, col_count, r_k, r_k_this, r_k_1, r_k_2, r_k_3);
               
                MPI_Send(r_k.data(), r_k.size(), MPI_DOUBLE, 1, 3, MPI_COMM_WORLD);
                MPI_Send(r_k.data(), r_k.size(), MPI_DOUBLE, 2, 3, MPI_COMM_WORLD);
                MPI_Send(r_k.data(), r_k.size(), MPI_DOUBLE, 3, 3, MPI_COMM_WORLD);
            }
            else {
                MPI_Send(r_k_this.data(), r_k_this.size(), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                MPI_Recv(r_k.data(), r_k.size(), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &status);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            mult(A, r_k, A_r_k_this, grid_size);
            if (nrank == 0) {
                vector<double> A_r_k_1(sizes[1], 0.0), A_r_k_2(sizes[2], 0.0), A_r_k_3(sizes[3], 0.0);
                MPI_Recv(A_r_k_1.data(), sizes[1], MPI_DOUBLE, 1, 5, MPI_COMM_WORLD, &status);
                MPI_Recv(A_r_k_2.data(), sizes[2], MPI_DOUBLE, 2, 5, MPI_COMM_WORLD, &status);
                MPI_Recv(A_r_k_3.data(), sizes[3], MPI_DOUBLE, 3, 5, MPI_COMM_WORLD, &status);

                build_vector_4(row_count, col_count, A_r_k, A_r_k_this, A_r_k_1, A_r_k_2, A_r_k_3);
                MPI_Send(A_r_k.data(), A_r_k.size(), MPI_DOUBLE, 1, 6, MPI_COMM_WORLD);
                MPI_Send(A_r_k.data(), A_r_k.size(), MPI_DOUBLE, 2, 6, MPI_COMM_WORLD);
                MPI_Send(A_r_k.data(), A_r_k.size(), MPI_DOUBLE, 3, 6, MPI_COMM_WORLD);
            }
            else {
                MPI_Send(A_r_k_this.data(), A_r_k_this.size(), MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
                MPI_Recv(A_r_k.data(), A_r_k.size(), MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
            }
            if (nrank == 0) {
                A_r_k_norm = get_norm(A_r_k, h_1, h_2);
                MPI_Recv(&scalar, 1, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, &status);
            }
            else if (nrank == 1){
                scalar = get_scalar(A_r_k, r_k, h_1, h_2);
                MPI_Send(&scalar, 1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (nrank == 0) {
                tau = scalar / (A_r_k_norm * A_r_k_norm);
                multiply(r_k, tau);
                sub(w_k, r_k, w_k_1);
                sub(w_k_1, w_k, w_k_1_w_k);
                d = get_norm(w_k_1_w_k, h_1, h_2);
                if (counter % 500 == 0) {
                    cout << d << endl;
                }
                if (d < delta) {
                    uint8_t flag = 1;
                    MPI_Send(&flag, 1, MPI_UINT8_T, 1, 7, MPI_COMM_WORLD);
                    MPI_Send(&flag, 1, MPI_UINT8_T, 2, 7, MPI_COMM_WORLD);
                    MPI_Send(&flag, 1, MPI_UINT8_T, 3, 7, MPI_COMM_WORLD);
                    cout << counter << endl;
                    for (int i = 0; i < w_k_1.size(); i++)
                        result[i] = w_k_1[i];
                    break;
                }
                else {
                    uint8_t flag = 0;
                    MPI_Send(&flag, 1, MPI_UINT8_T, 1, 7, MPI_COMM_WORLD);
                    MPI_Send(&flag, 1, MPI_UINT8_T, 2, 7, MPI_COMM_WORLD);
                    MPI_Send(&flag, 1, MPI_UINT8_T, 3, 7, MPI_COMM_WORLD);

                    MPI_Send(w_k_1.data(), w_k_1.size(), MPI_DOUBLE, 1, 8, MPI_COMM_WORLD);
                    MPI_Send(w_k_1.data(), w_k_1.size(), MPI_DOUBLE, 2, 8, MPI_COMM_WORLD);
                    MPI_Send(w_k_1.data(), w_k_1.size(), MPI_DOUBLE, 3, 8, MPI_COMM_WORLD);

                    w_k = w_k_1;
                }
            }
            else {
                uint8_t flag;
                MPI_Recv(&flag, 1, MPI_UINT8_T, 0, 7, MPI_COMM_WORLD, &status);
                if (flag)
                    break;
                else
                    MPI_Recv(w_k.data(), w_k.size(), MPI_DOUBLE, 0, 8, MPI_COMM_WORLD, &status);
            }
            counter++;
        } while (true);
        if (nrank == 0) {
            auto diff = chrono::steady_clock::now() - start;
            cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
        }
        for (int i = 0; i < A.size(); i++) {
            delete A[i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        if (nrank == 0) {
            ofstream output("result_MPI.txt");
            for (int i = 0; i < grid_size + 1; i++) {
                for (int j = 0; j < grid_size + 1; j++) {
                    output << result[i * (grid_size + 1) + j] << " ";
                }
                output << endl;
            }
        }
    }
    return 0;
}