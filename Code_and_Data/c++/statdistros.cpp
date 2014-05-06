
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cmath>
#include <cstring>

#include <stdexcept>

#include <mkl.h>
#include <mkl_lapack.h>
#include <mkl_blas.h>

#include "statdistros.hpp"


std::vector<RngStream> MyRngStream::AllStreams;

void MyRngStream::RngS_set_random_seed()
{
    unsigned long seed[6] = {0, 0, 0, 0, 0, 0};
    srand(time(NULL));
    while( seed[0] + seed[1] + seed[2] == 0 ) {
        seed[0] = rand() % 4294967087;
        seed[1] = rand() % 4294967087;
        seed[2] = rand() % 4294967087;
    }
    while( seed[3] + seed[4] + seed[5] == 0 ) {
        seed[3] = rand() % 4294944443;
        seed[4] = rand() % 4294944443;
        seed[5] = rand() % 4294944443;
    }
    RngStream_SetPackageSeed(seed);
}

void MyRngStream::ClearAllStreams()
{
//     for(std::vector<RngStream>::iterator s=AllStreams.begin(); s != AllStreams.end(); ++s)
    for(auto s=AllStreams.begin(); s != AllStreams.end(); ++s)
            RngStream_DeleteStream(&(*s));
    AllStreams.clear();
}

/************************************************************************
 *     Categorical                                                      *
 ************************************************************************/

Categorical::Categorical(int k, const double *p, int Idx)
: Distribution<int>(Idx)
{
    num_cat = k;
    prob = new double[num_cat];
    cum_prob = new double[num_cat];
    memcpy(prob, p, k*sizeof(double));
    cum_prob[0] = prob[0];
    assert(prob[0] >= 0.0);
    for(int i=1; i<k; ++i) {
        assert(prob[i] >= 0.0);
        cum_prob[i] = cum_prob[i-1] + prob[i];
    }
    if (fabs(cum_prob[k-1]-1.0) > 1e-14) {
        for(int i=0; i<k; ++i) {
            prob[i] /= cum_prob[k-1];
            cum_prob[i] /= cum_prob[k-1];
        }
    }
}

int Categorical::rand() const
{
    double U = G.uRand();
    int i;
    for(i=0; U > cum_prob[i]; ++i) {}
    return i;
}

/************************************************************************
 *     Gaussian                                                         *
 ************************************************************************/

void Gaussian::Box_Muller(double &u, double &v) const
{
    /** this is the polar form of the Box-Muller transform.  It produces
     * two standard normal values from two (0,1) uniform values, but only
     * if they fall inside the unit circle.  Even though it throws away
     * approximately 28% of the random numbers, it is still faster than the
     * regular form of the Box-Muller transform, because it eliminates the
     * evaluation of sin and cos functions
     **/
    double s;
    do {
        u = G.uRand()*2.0 - 1.0;
        v = G.uRand()*2.0 - 1.0;
        s = u*u + v*v;
    } while( (s>=1.0) || (s==0.0) );
    s = sqrt( -2.0*log(s)/s );
    u *= s;
    v *= s;
}

double Gaussian::rand() const
{
    double ret;
    if( isnan(saved) ) {
        Box_Muller(ret, saved);
    } else {
        ret = saved;
        saved = NAN;
    }
    if(standard) return ret;
    else return mean + stddev * ret;
}

double Gaussian::logpdf(double pt) const
{
    pt = pt - mean;
    if(stddev==0.0) { /* comaring floating numbers for equality !!! */
        if(pt==0.0) return INFINITY;
        else return -INFINITY;
    }
    double var = stddev * stddev;
    return -0.5*(log(2.0*M_PI*var)+pt*pt/var);
}

void Gaussian::sample(int n, double pts[]) const
{
    int i = 0; /** start with i=0 */
    if( !isnan(saved) ) {
        pts[i] = saved;
        saved = NAN;
        ++i;
    }
    for(; i+1<n; i+=2)
        Box_Muller(pts[i], pts[i+1]);
    if(i<n) Box_Muller(pts[i], saved);
    if( !standard ) {
        for(i=0; i<n; ++i) {
            #ifdef FP_FAST_FMA
            pts[i] = fma(stddev, pts[i], mean);
            #else
            pts[i] = stddev*pts[i] + mean;
            #endif
        }
    }
}

void Gaussian::sample_logpdf(int n, const double pts[], double vals[]) const
{
    if(stddev==0.0) { /* comaring floating numbers for equality !!! */
        for(int i=0; i<n; ++i) {
            double pt = pts[i]-mean;
            if(pt==0.0) vals[i]=INFINITY;
            else vals[i] = -INFINITY;
        }
    } else {
        double aux = log(2.0*M_PI*stddev*stddev);
        for(int i=0; i<n; ++i) {
            double stdpt = (pts[i]-mean) / stddev;
            #ifdef FP_FAST_FMA
            vals[i] = -0.5*fma(stdpt,stdpt,aux);
            #else
            vals[i] = -0.5*(stdpt*stdpt + aux);
            #endif
        }
    }
}

void Gaussian::sample_pdf(int n, const double pts[], double vals[]) const
{
    sample_logpdf(n, pts, vals);
    for(int i=0; i<n; ++i)
        vals[i] = exp(vals[i]);
}

/************************************************************************
 *     Multivariate Gaussian                                            *
 ************************************************************************/
/** NOTES:
 *    - all matrices are stored column-major, i.e. the Fortran way
 *    - deviates are row-vectors, e.g. sample_rand() returns a
 *       matrix with n rows and dim columns, where each row is a random point
 *    - mean must be a vector of length dim
 *    - the correlation structure is given by the square-root of the variance
 *      matrix. This can be obtained by calling SqrtMatrix.
 *    - cov/sigma must be a dim-by-dim s.p.d. matrix, sqrt_cov_mat is the
 *      upper-triangular Cholesky decomposition, s.t.
 *           transpose(sqrt_cov_mat) * sqrt_cov_mat = cov
 *    - we do not allocate/deallocate memory, user must manage their own RAM
 *
 **/

MultivariateGaussian::MultivariateGaussian(int d, int Idx)
    : Gaussian(Idx), standard_mv(true),
        mean_vec(NULL), sqrt_cov_mat(NULL), dim(d)
{}

MultivariateGaussian::MultivariateGaussian(int d, const double *mu,
                const double *sigma, int Idx)
    : Gaussian(Idx), standard_mv(false),
        mean_vec(NULL), sqrt_cov_mat(NULL), dim(d)
{
    new_mean_cov();
    SetCovariance(sigma);
    SetMean(mu);
}

void MultivariateGaussian::SetStandardMeanCovariance()
{
    if (!standard_mv) {
        standard_mv = true;
        delete_mean_cov();
    }
}

void MultivariateGaussian::SetStandardMeanScaleCovariance(double scale)
{
    if (standard_mv) {
        standard_mv = false;
        new_mean_cov();
        memset(mean_vec, 0, dim*sizeof(double));
    }
    assert(scale > 0.0);
    DiagonalMatrix(sqrt_cov_mat, sqrt(scale));
}

void MultivariateGaussian::SetMean(const double *mu)
{
    if (standard_mv) {
        standard_mv = false;
        new_mean_cov();
        IdentityMatrix(sqrt_cov_mat);
    }
    memcpy(mean_vec, mu, dim*sizeof(double));
}

void MultivariateGaussian::SetCovariance(const double *cov)
{
    if (standard_mv) {
        standard_mv = false;
        new_mean_cov();
        memset(mean_vec, 0, dim*sizeof(double));
    }
    memcpy(sqrt_cov_mat, cov, dim*dim*sizeof(double));
    SqrtMatrix(sqrt_cov_mat);
}

void MultivariateGaussian::SqrtMatrix(double *mat) const
throw (std::invalid_argument)
{
    int info;
    dpotrf("U", &dim, mat, &dim, &info);
    if(info != 0) {
        std::invalid_argument e("Matrix is not positive-definite");
        throw e;
    }
    /** set below the main diagonal to zero */
    for(int i=1, j=1; j<dim; i+=dim+1, ++j)
        memset(mat+i, 0, sizeof(double)*(dim-j));
}

void MultivariateGaussian::DiagonalMatrix(double *mat, double alpha) const
{
    memset(mat, 0, dim*dim*sizeof(double));
    for(int i=0; i<dim*dim; i+=dim+1) mat[i]=alpha;
}

void MultivariateGaussian::DiagonalMatrix(double *mat, const double *diag) const
{
    int i_1=1, i_dimp1=dim+1;
    memset(mat, 0, dim*dim*sizeof(double));
    dcopy(&dim, diag, &i_1, mat, &i_dimp1);
}

void MultivariateGaussian::sample(int n, double pts[]) const
{
    stdnorm::sample(n*dim, pts);
    if(standard_mv) {
        return;  /** this was easy **/
    }
    /** adjust the mean and the covariance.
     *  we need to do
     *        pts = mean_vec + pts * sqrt_cov_mat
     *  pts is n-by-dim
     *  mean_vec is 1-by-dim, so we need singleton expansion
     *  sqrt_cov_mat is dim-by-dim
     **/
    /** call blas function to do TRiangular Matrix-Matrix multiplication
     *        pts = pts * sqrt_cov_mat
     **/
    int in = n;  /** our blas uses 32-bit signed integers **/
    double one = 1.0;
    dtrmm("R", "U", "N", "N", &in, &dim, &one, sqrt_cov_mat, &dim, pts, &in);
    /** last, add the mean:   pts(i,:) = pts(i,:) + mean  for each row i */
    for(int j=0; j<dim; ++j)
        for(int i=0; i<n; ++i)
            pts[i+j*n] += mean_vec[j];
}

void MultivariateGaussian::sample_logpdf(int n,
        const double pts[], double vals[]) const
{
    const int i_one=1;
    const double one_d = 1.0, mone_d = -1.0;
    double a = -0.5*log(2.0*M_PI)*dim;
    if (!standard_mv) {
        for(int i=0; i<dim*dim; i+=dim+1)
            a -= log(sqrt_cov_mat[i]);
    }
    /** compute logpdf(i)=a-0.5*(x(i,:)-mean)*inv(cov)*(x(i,:)-mean)' */
    double b = 0.0;
    if (!standard_mv) {
        double *tmp = new double[dim];
        for(int i=0; i<n; ++i)  {
            /** tmp = pts(i,:) **/
            dcopy(&dim, pts+i, &n, tmp, &i_one);
            /** tmp =  -1.0*mean_vec+tmp **/
            daxpy(&dim, &mone_d, mean_vec, &i_one, tmp, &i_one);
            /** solve Y*sqrt_var=tmp for Y
              * the answer goes back into tmp, so there is no Y **/
            dtrsm("R", "U", "N", "N", &i_one, &dim, &one_d, sqrt_cov_mat,
                &dim, tmp, &i_one);
            b = ddot(&dim, tmp, &i_one, tmp, &i_one);
            vals[i] = a-0.5*b;
        }
        delete[] tmp;
    } else {
        /** no need to deal with mean and cov, so b = x(i,:)*x(i,:)' */
        for(int i=0; i<n; ++i)  {
            b = ddot(&dim, pts+i, &n, pts+i, &n);
            vals[i] = a-0.5*b;
        }
    }

}

/************************************************************************
 *     Mixture of Multivariate Gaussian                                 *
 ************************************************************************/


void MixtureMVN::sample(int n, double pts[]) const
{
    int i_one=1;
    double *tmp = new double[dim];
    for(int i=0; i<n; ++i) {
        int bin = C.rand();
        MVvec[bin].sample(1, tmp);
        dcopy(&dim, tmp, &i_one, pts+i, &n);
    }
    delete[] tmp;
}

void MixtureMVN::sample_logpdf(int n, const double pts[], double vals[]) const
{
    sample_pdf(n, pts, vals);
    for(int i=0; i<n; ++i)
        vals[i] = log(vals[i]);
}

void MixtureMVN::sample_pdf(int n, const double pts[], double vals[]) const
{
    int i_one=1;
    double *tmp = new double[n];
    memset(vals, 0, n*sizeof(double));
    for(int bin=0; bin<C.num_cat; ++bin) {
        double pr = C.pdf(bin);
        MVvec[bin].sample_pdf(n, pts, tmp);
        daxpy(&n, &pr, tmp, &i_one, vals, &i_one);
    }
    delete[] tmp;
}




