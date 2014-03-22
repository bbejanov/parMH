

if(!is.loaded("n_rand")) dyn.load("libstatdist.so")

nstd_rand <- function(n, ..., w=0) {
    .C("nstd_rand", as.integer(w), as.integer(n), 
    vals=as.double(rep(0,n)), DUP=FALSE)$vals
}

n_rand <- function(n, mean=0, var=1, ..., w=0) {
    .C("nstd_rand", as.integer(w), 
        as.integer(n), 
        vals=as.double(rep(0,n)), 
        as.double(mean), 
        as.double(var),    
        DUP=FALSE
    )$vals    
}


n_logpdf <- function(x, mean=0, var=1) {
    n <- length(x)
    k <- length(mean)
    .C("n_logpdf", as.integer(n), as.double(x), 
        val=as.double(rep(0,n)), 
        as.integer(k), as.double(mean), as.double(var), 
        DUP=FALSE
    )$val
}

n_pdf <- function(x, mean=0, var=1) {
    n <- length(x)
    .C("n_pdf", as.integer(n), as.double(x), 
        val=as.double(rep(0,n)), 
        as.double(mean), as.double(var), 
        DUP=FALSE
    )$val
}

n <- 5000000;
print(system.time(
    cc <- nstd_rand(n)
))
print(summary(cc))
# chi-squared test
chi2 = sum(cc^2)

# qqplot(cc, rnorm(n));

print(system.time(
    dd <- n_pdf(cc)
))
print(err <- max(abs(dd-dnorm(cc))))
stopifnot(err < 1e-14)


cat("==============================================\n")

{
print(var <- matrix(c(4,-1,0,-1,2,-0.5,0,-0.5,7),3,3))
k <- nrow(var)
print(sqv <- matrix(.C("mvn_sqrt", as.integer(k), as.double(var), 
            as.double(rep(0,k*k)), DUP=FALSE)[[3]],k,k))
print(mean <- matrix(c(-1,5,0),1,3))
n <- 50000
ret <- .C("mvn_rand", w=as.integer(0), n=as.integer(n), 
            rv=as.double(rep(0,n*k)), dim=as.integer(k), 
            mean=as.double(mean), var=as.double(var), DUP=FALSE)

QQ <- matrix(ret$rv, n,k)
print(colMeans(QQ))
print(cov(QQ))
}

cat("==============================================\n")

print(system.time(ret <- .C("mvn_logpdf", as.integer(n), as.double(QQ), 
            lp=as.double(rep(0,n)), as.integer(k), as.double(mean), 
            as.double(var), DUP=FALSE))
)


require(mvtnorm)
foo <- dmvnorm(QQ, mean, var, log=TRUE)

print(err <- max(abs(foo-ret$lp)))
stopifnot(err < 1e-14)

cat("==============================================\n")

n <- 2000
pr <- c(0.8, 0.2)
mu <- c(-3, 0, 1, 4)
vars <- c(1, 0, 0, 1, 4, 0, 0, 4)
sqv <- sqrt(vars)
ret <- .C("mvnm_rand", w=as.integer(0), n=as.integer(n), vals=as.double(rep(0,2*n)), k=as.integer(length(pr)), as.double(pr), as.integer(length(mu)/length(pr)), as.double(mu), as.double(sqv), DUP=FALSE)
QQ <- matrix(ret$vals, n, 2)
plot(QQ[,1], QQ[,2])
qqplot(QQ[,1], QQ[,2])
hist(QQ[,1], breaks=50)
hist(QQ[,2], breaks=50)

ret <- .C("mvnm_pdf", n=as.integer(n), as.double(QQ), p=as.double(rep(0,n)), as.integer(length(pr)), as.double(pr), as.integer(length(mu)/length(pr)), as.double(mu), as.double(sqv), DUP=FALSE)
write.csv(data.frame(QQ,ret$p), file="surface.csv")
cat('scatter-plot the data in surface.csv to see that it makes sense\n')



