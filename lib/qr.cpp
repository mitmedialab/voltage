#include <math.h>
#include "wrapper.hpp"


// calculate Qx where Q = I - 2 v v^T
//
// instead of first constructing matrix Q and then computing Qx,
// it is much faster to do
// Qx = (I - 2 v v^T)x = Ix - 2 v v^T x = x - 2(v^T x)v
static void householder_transform_vec(int n, double *v, double *x)
{
    double v_dot_x = 0;
    for(int i = 0; i < n; i++)
    {
        v_dot_x += v[i] * x[i];
    }
    for(int i = 0; i < n; i++)
    {
        x[i] -= 2 * v_dot_x * v[i];
    }
}

// calculate QA and Qb where Q = I - 2 v v^T
static void householder_transform(int row, int col, double *v, double *a)
{
    //#pragma omp parallel for
    for(int i = 0; i <= row; i++)
    {
        householder_transform_vec(col, v, a+i);
    }
}

// Solve min |A x = b|^2 with QR decomposition using Householder method
//
// A is a row x col matrix where row > col, and it is
// decomposed A as A = QR where Q is an orthogonal square matrix (row x row)
// and R is an upper triangular matrix (row x col but most elements are zero)
//
// The problem is translated into min |QR x = b|^2
// and by left-multiplying Q^T, it further becomes |R x = Q^T b|^2
// Indeed, the upper triangular R is constructed as Q^T A
// Therefore, we do not need to maintain Q but via Householder transform
// we can progressively turn A and b into Q^T A (= R) and Q^T b
// The problem min |R x = Q^T b|^2 can be solved by back substitution
// and the residual is the bottom (row - col) part of |Q^T b|^2
//
// !!!!! CAUTION !!!!!
// For implementation reason, a[i][j] represents j-th row and i-th column
// and a[-1][j] represents b[j], j-th row of b
double solve_with_qr_decomposition(int row, int col, double *a, double *x)
{
    double *v = (double *) malloc(row * sizeof(double));
    int m = row;
    int n = col;   
    // work on the lower right submatrix of A whose top left is (i, i)
    // to progressively transform A into an upper triangular
    for(int i = 0; i < row; i++)
    {
        // pick the i-th column from the submatrix
        double sum = 0;
        for(int j = 0; j < col; j++)
        {
            v[j] = (j < i) ? 0 : *loc2D(a, m, (n+1), i, j);
            sum += v[j] * v[j];
        }
        double alpha = sqrt(sum);
        // opposite sign to the i-th element of the chosen vector
        if(v[i] > 0) alpha = -alpha;
        
        // make a vector for Householder reflection
        sum = 0;
        for(int j = 0; j < col; j++)
        {
            v[j] -= (j == i) ? alpha : 0;
            sum += v[j] * v[j];
        }
        double beta = sqrt(sum);
        
        //#pragma omp parallel for
        for(int j = 0; j < col; j++)
        {
            v[j] /= beta;
        }
        
        // Householder transform Qi A and Qi b
        // where Qi = I - 2 v v^T
        householder_transform(row, col, v, a);
    }
    // Q^T = Qcol ... Q2 Q1
    // therefore, now A <-- Q^T A (= R) and b <-- Q^T b
    
    // back substitution as A is now an upper triangular
    for(int i = col - 1; i >= 0; i--)
    {
        x[i] = *loc(a, m, (n+1), i, col);
        for(int j = i + 1; j < col; j++)
        {
            x[i] -= *loc(a, m, (n+1), i, j) * x[j];
        }
        x[i] /= *loc(a, m, (n+1), i, i);
    }
    
    free_double1d(v);
    
    return 0;
}

