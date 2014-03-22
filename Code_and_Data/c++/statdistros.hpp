#ifndef __STATDISTROS_H
#define __STATDISTROS_H

extern "C" {
#include <RngStream.h>
}

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <array>
#include <stdexcept>


/** this class handles independent random streams as per RngStream pachage */
class MyRngStream
{
    /** we keep a static vector (AllStreamse) of all streams ever created;
     *  streams are referenced by Index within this vector;
     *  this allows the same stream ot be used by multiple generators.
     *  An instace is initialized from an index.
     *  The static vector is extended as necessary.
     *  The stream data member is a pointer to one of AllStreams elements
     */
protected:    
    static std::vector<RngStream> AllStreams;
    RngStream *stream;  /** points to one of the elements of AllStreams */
    
    static RngStream newRngStream()
    {
        RngStream st = RngStream_CreateStream("");
        /** increase precision of stream for generating double precision */
        RngStream_IncreasedPrecis(st, 1);
        return st;
    }    

public:
    explicit MyRngStream(size_t Idx = 0) {
        while (AllStreams.size() <= Idx) {
            /** extend AllStreams as necessary */
            AllStreams.push_back(newRngStream());
        }
        stream = &AllStreams[Idx];
    }

    static void RngS_set_random_seed();
    
    static void ClearAllStreams();  /** there is no need to ever call this */
 
    /** just for testing/debugging */
    static size_t CountStreams() { return AllStreams.size(); }

    /** user-defined conversion to RngStream */
    operator RngStream() { return *stream; }

    /** uniform distribution on [0,1] */
    double uRand() { return RngStream_RandU01(*stream); }

};

/************************************************************************/

/** the base class for all distributions */
template <class Point> 
class Distribution
{
protected:
    mutable MyRngStream G;
public:
    explicit Distribution<Point>(size_t Idx=0) : G(Idx) {}
    explicit Distribution<Point>(MyRngStream &g) : G(g) {}

    /** abstract methods -- one point at a time */
    virtual Point rand() const = 0;
    virtual double pdf(Point pt) const = 0;
    virtual double logpdf(Point pt) const = 0;

    /** vector-equivalent -- work on entire sample of size n */
    /** please redefine these in your derived class */
    virtual void sample(size_t n, Point pts[]) const
    {
        for(size_t i=0; i<n; ++i)
            pts[i] = rand();
    }
    virtual void sample_pdf(size_t n, const Point pts[], double vals[]) const
    {
        for(size_t i=0; i<n; ++i)
            vals[i] = pdf(pts[i]);
    }
    virtual void sample_logpdf(size_t n, const Point pts[], double vals[]) const
    {
        for(size_t i=0; i<n; ++i)
            vals[i] = logpdf(pts[i]);
    }
};

/************************************************************************/

class Uniform : public Distribution<double>
{
private:
    double left, width;
public:
    Uniform(double a=0.0, double b=0.0, size_t Idx=0)
    : Distribution<double>(Idx), left(a), width(b-a) {}
    
    virtual double rand() const
    { return left + width * G.uRand(); }

    virtual double pdf(double pt) const
    { return 1.0/width; }

    virtual double logpdf(double pt) const
    { return log(1.0/width); }
};

/************************************************************************/

class Categorical : public Distribution<int>
{
private:
    size_t num_cat;   /** number of categories */
    double *prob;     /** array of probabilities */
    double *cum_prob; /** array of cummulative probabilities */
public:
    Categorical(size_t k, const double *p, size_t Idx=0);
    virtual ~Categorical() { delete[] prob; delete[] cum_prob; }
    virtual int rand() const;
    virtual double pdf(int pt) const
    { assert(pt<num_cat); return prob[pt]; }
    virtual double logpdf(int pt) const
    { assert(pt<num_cat); return log(prob[pt]); }

    Categorical(const Categorical &C) : Distribution<int>(C) {
        num_cat = C.num_cat;
        prob = new double[num_cat];
        memcpy(prob, C.prob, num_cat*sizeof(double));
        cum_prob = new double[num_cat];
        memcpy(cum_prob, C.cum_prob, num_cat*sizeof(double));
    }

    Categorical &operator=(const Categorical &C) {
        Distribution<int>::operator=(C);
        num_cat = C.num_cat;
        prob = new double[num_cat];
        memcpy(prob, C.prob, num_cat*sizeof(double));
        cum_prob = new double[num_cat];
        memcpy(cum_prob, C.cum_prob, num_cat*sizeof(double));
        return *this;
    }
        
    friend class MixtureMVN;
};

/************************************************************************/

class Gaussian : public Distribution<double>
{
private:
    double mean;
    double stddev;
    bool standard;
    mutable double saved;
    void Box_Muller(double &u, double &v) const;
public:
    explicit Gaussian(size_t Idx=0)
        : Distribution<double>(Idx), mean(0.0), stddev(1.0),
        standard(true), saved(NAN)
        {}
        
    Gaussian(double m, double sd, size_t Idx=0)
        : Distribution<double>(Idx), mean(m), stddev(sd), standard(false), saved(NAN)
        {assert(stddev>=0.0);}
    virtual double rand() const;
    virtual double pdf(double pt) const { return exp(logpdf(pt)); }
    virtual double logpdf(double pt) const;
    virtual void sample(size_t n, double pts[]) const;
    virtual void sample_logpdf(size_t n, const double pts[], double vals[]) const;
    virtual void sample_pdf(size_t n, const double pts[], double vals[]) const;
};

/************************************************************************/

class MultivariateGaussian : public Gaussian
{
    /** this class derives from Gaussian and always instantiates its
     *  Gaussian parts to a standard univariate Gaussian.
     *  The parent sample() is called within this sampe() to
     *  get n*dim standard normals, which are then transformed to the
     *  desired multivatiate
     **/
private:
    typedef Gaussian stdnorm;

    /* do not call this function, call sample(1, pts) instead */
    virtual double rand() const {
        throw new std::logic_error("Don't use MultivariateGaussian::rand()");
    }
    virtual double pdf(double pt) const {
        throw new std::logic_error("Don't use MultivariateGaussian::pdf()");
    }
    virtual double logpdf(double pt) const { 
        throw new std::logic_error("Don't use MultivariateGaussian::logpdf()");
    }
protected:
    int dim;
    double *mean_vec;
    double *sqrt_cov_mat;
    bool standard_mv;

    void new_mean_cov()    {
        mean_vec = new double[dim];
        sqrt_cov_mat = new double[dim*dim];
    }
    void delete_mean_cov() {
        delete[] mean_vec;
        delete[] sqrt_cov_mat;
    }
public:
    virtual ~MultivariateGaussian(){ if (!standard_mv) delete_mean_cov(); }
    explicit MultivariateGaussian(size_t d, size_t Idx=0);
    MultivariateGaussian(size_t d, const double *mu,
                    const double *sigma_half, size_t Idx=0);

    MultivariateGaussian(const MultivariateGaussian &MV)
        : Gaussian(MV), mean_vec(NULL), sqrt_cov_mat(NULL)
    {
        dim=MV.dim;
        standard_mv=MV.standard_mv;
        if(!standard_mv) {
            new_mean_cov();            
            memcpy(mean_vec, MV.mean_vec, dim*sizeof(double));
            memcpy(sqrt_cov_mat, MV.sqrt_cov_mat, dim*dim*sizeof(double));
        }
    }
    MultivariateGaussian &operator =(const MultivariateGaussian &MV) {
        if(!standard_mv) {
            delete_mean_cov();
        }
        Gaussian::operator =(MV);
        dim=MV.dim;
        standard_mv=MV.standard_mv;
        if(!standard_mv) {
            new_mean_cov();            
            memcpy(mean_vec, MV.mean_vec, dim*sizeof(double));
            memcpy(sqrt_cov_mat, MV.sqrt_cov_mat, dim*dim*sizeof(double));
        }
        return *this;
    }

    void SetStandardMeanCovariance();
    void SetMean(const double *mu);
    void SetCovariance(const double *cov);
    
    void SqrtMatrix(double *mat) const throw (std::invalid_argument);
    void IdentityMatrix(double *mat) const;

    /** The pts[] array must be of length n*dim.  The points are stored
     *  as a n-by-dim matrix in column major order, i.e. the first n
     *  doubles are the first marginal, the next n are the second marginal, etc.
     *  The non-sample versions are disabled because we don't want to manage
     *  the caller's memory; use sample_xyz with n=1 instead
     */
    virtual void sample(size_t n, double pts[]) const;
    virtual void sample_logpdf(size_t n, const double pts[], double vals[]) const;

    friend class MixtureMVN;
};

/************************************************************************/


class MixtureMVN : public Distribution<double>
{
private:
    /* do not call this function, call sample(1, pts) instead */
    virtual double rand() const {
        throw new std::logic_error("Don't use MixtureMVN::rand()");
    }
    virtual double pdf(double pt) const {
        throw new std::logic_error("Don't use MixtureMVN::pdf()");
    }
    virtual double logpdf(double pt) const { 
        throw new std::logic_error("Don't use MixtureMVN::logpdf()");
    }
protected:
    int dim;
    Categorical C;
    std::vector<MultivariateGaussian> MVvec;
public:
    MixtureMVN(size_t k, const double *p,
            const std::vector<MultivariateGaussian> &MV, size_t Idx=0)
        : C(k, p, Idx), MVvec(MV)
        {
            assert(MVvec.size() == k);
            dim = MVvec[0].dim;
            for(size_t i=1; i<k; ++i)
                assert(MVvec[i].dim == dim);
        }

    virtual void sample(size_t n, double pts[]) const;
    virtual void sample_logpdf(size_t n, const double pts[], double vals[]) const;
    virtual void sample_pdf(size_t n, const double pts[], double vals[]) const;
};






#endif
