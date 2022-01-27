# Causal Inference Workshop - 9 September 2021

library(rethinking)

####################################################
# TWO MOMS

# first without confound
set.seed(1908)
N <- 200 # number of pairs
# birth order and family sizes
B1 <- rbinom(N,size=1,prob=0.5) # 50% first borns
M <- rnorm( N , 2*B1 )
B2 <- rbinom(N,size=1,prob=0.5)
D <- rnorm( N , 2*B2 + 0*M ) # change the 0 to turn on causal influence of mom

# model with B1 has worse precision, model with B2 has better precision
# what is going on here?
summary( lm( D ~ M ) )
summary( lm( D ~ M + B1 ) )
summary( lm( D ~ M + B2 ) )

plot( coeftab( lm( D ~ M ) , lm( D ~ M + B1 ) ,  lm( D ~ M + B2 ) ) , pars="M" )

# now with confound, it gets worse for B1
set.seed(1908)
N <- 200 # number of pairs
U <- rnorm(N) # simulate confounds
# birth order and family sizes
B1 <- rbinom(N,size=1,prob=0.5) # 50% first borns
M <- rnorm( N , 2*B1 + U )
B2 <- rbinom(N,size=1,prob=0.5)
D <- rnorm( N , 2*B2 + U + 0*M ) # change the 0 to turn on causal influence of mom

# fit the two regression models
summary( lm( D ~ M ) )
summary( lm( D ~ M + B1 + B2 ) )

plot( coeftab( lm( D ~ M ) , lm( D ~ M + B1 ) ,  lm( D ~ M + B2 ) ) , pars="M" )

# compare the models with AIC
AIC( lm( D ~ M ) )
AIC( lm( D ~ M + B1 ) )

# full-luxury bayesian inference

# best case scenario, if we observed confound
precis( lm( D ~ M + B2 + U ) )

library(rethinking)
library(cmdstanr)
dat <- list(N=N,M=M,D=D,B1=B1,B2=B2)
set.seed(1908)
flbi <- ulam(
    alist(
        # mom model
            M ~ normal( mu , sigma ),
            mu <- a1 + b*B1 + k*U[i],
        # daughter model
            D ~ normal( nu , tau ),
            nu <- a2 + b*B2 + m*M + k*U[i],
        # B1 and B2
            B1 ~ bernoulli(p),
            B2 ~ bernoulli(p),
        # unmeasured confound
            vector[N]:U ~ normal(0,1),
        # priors
            c(a1,a2,b,m) ~ normal( 0 , 0.5 ),
            c(k,sigma,tau) ~ exponential( 1 ),
            p ~ beta(2,2)
    ), data=dat , chains=4 , cores=4 , iter=2000 , cmdstan=TRUE )

precis(flbi)

m <- M
plot( coeftab( lm( D ~ m ) , lm( D ~ m + B1 ) ,  lm( D ~ m + B2 ) , flbi ) , pars="m" )

post <- extract.samples(flbi)
Uest <- apply(post$U,2,mean)
blank()
plot(U,Uest,xlab="U (simulated)",ylab="U (estimated)", col=2 , lwd=2 )
abline(a=0,b=1,lty=2)

# version that marginalizes out the missing data
flbi_plus <- ulam(
    alist(
        c(M,D) ~ multi_normal( c(mu,nu) , Rho , Sigma ),
        mu <- a1 + b*B1,
        nu <- a2 + b*B2 + m*M,
        c(a1,a2,b,m) ~ normal( 0 , 0.5 ),
        Rho ~ lkj_corr( 2 ),
        Sigma ~ exponential( 1 )
    ), data=dat , chains=4 , cores=4 , cmdstan=TRUE )

precis(flbi_plus,3)

# more exotic example - no instrument (B1 -> D), but have two measures of U

set.seed(1908)
N <- 200 # number of pairs
U <- rnorm(N,0,1) # simulate confound
V <- rnorm(N,U,1)
W <- rnorm(N,U,1)
# birth order and family sizes
B1 <- rbinom(N,size=1,prob=0.5) # 50% first borns
M <- rnorm( N , 2*B1 + U )
B2 <- rbinom(N,size=1,prob=0.5)
D <- rnorm( N , 2*B2 + 0.5*B1 + U + 0*M )

# confounded regression
precis( lm( D ~ M + B1 + B2 + V + W ) )

# full-luxury bayesian inference
dat2 <- list(N=N,M=M,D=D,B1=B1,B2=B2,V=V,W=W)
flbi2 <- ulam(
    alist(
        M ~ normal( muM , sigmaM ),
        muM <- a1 + b*B1 + k*U[i],
        D ~ normal( muD , sigmaD ),
        muD <- a2 + b*B2 + d*B1 + m*M + k*U[i],
        W ~ normal( muW , sigmaW ),
        muW <- a3 + w*U[i],
        V ~ normal( muV , sigmaV ),
        muV <- a4 + v*U[i],
        vector[N]:U ~ normal(0,1),
        c(a1,a2,a3,a4,b,d,m) ~ normal( 0 , 0.5 ),
        c(k,w,v) ~ exponential( 1 ),
        c(sigmaM,sigmaD,sigmaW,sigmaV) ~ exponential( 1 )
    ), data=dat2 , chains=4 , cores=4 , iter=2000 , cmdstan=TRUE )

precis(flbi2)

####################################################
# PEER BIAS

library(rethinking)

# simulation in which there is no discrimination
# conditioning on E reveals the truth
set.seed(1914)
N <- 500
X <- rbern(N,prob=0.5)
pY <- c( 0.25 , 0.05 )
pE <- X*inv_logit(-2) + (1-X)*inv_logit(+1)
E <- sapply( 1:N , function(n) sample( 1:2 , size=1 , prob=c(pE[n],1-pE[n]) ) )

p <- pY[E]
Y <- rbern(N,prob=p)

precis( glm( Y ~ X , family=binomial ) )

precis( glm( Y ~ X + E , family=binomial ) )

mg0 <- glm( Y ~ X , family=binomial )
mg1 <- glm( Y ~ X + E , family=binomial )
plot( coeftab( mg0 , mg1 ) , pars="X" )


# simulation in which there really is discrimination
# condition on E hides the truth
set.seed(1914)
set.seed(1964)
N <- 500
Q <- rnorm(N)
X <- rbern(N,prob=0.5)
pY <- c( 0.25 , 0.1 )
pE <- X*inv_logit(Q-2) + (1-X)*inv_logit(Q+1)
E <- sapply( 1:N , function(n) sample( 1:2 , size=1 , prob=c(pE[n],1-pE[n]) ) )

p <- inv_logit( logit(pY[E]) + Q - X )
Y <- rbern(N,prob=p)

precis( glm( Y ~ X , family=binomial ) )

precis( glm( Y ~ X + E , family=binomial ) )

precis( glm( Y ~ X + E + Q , family=binomial ) )

mg0 <- glm( Y ~ X , family=binomial )
mg1 <- glm( Y ~ X + E , family=binomial )
mg2 <- glm( Y ~ X + E + Q , family=binomial )
plot( coeftab( mg0 , mg1 ) , pars="X" )

# descendants of Q
R1 <- rnorm(N,0.5*Q)
R2 <- rnorm(N,0.5*Q)

mg3 <- glm( Y ~ X + E + R1 + R2 , family=binomial )
precis( mg3 )
plot( coeftab( mg0 , mg1 , mg3 ) , pars="X" )

# bayes model
dat <- list( Y=Y , E=E , XX=X , id=1:N , R1=R1 , R2=R2 )
mR <- ulam(
    alist(
        # Y model
        Y ~ bernoulli(p),
        logit(p) <- a[E] + X*XX + h*Q[id],
        a[E] ~ normal(0,1),
        X ~ normal(0,1),
        h ~ half_normal(0,1),
        # Q model
        vector[id]:Q ~ normal(0,1),
        R1 ~ normal(Q,1),
        R2 ~ normal(Q,1)
    ) , data=dat , chains=4 , cores=4 , cmdstan=TRUE )

precis(mR,2,omit="Q")

plot( coeftab( mg0 , mg1 , mg3 , mR ) , pars="X" )

post <- extract.samples(mR)
Qest <- apply(post$Q,2,mean)
blank()
plot(Q,Qest)
abline(a=0,b=1,lty=2)

# confounded but we do a partial identification analysis
# we use an informative prior for h (effect of Q)

dat2 <- list( Y=Y , E=E , X=X , id=1:N )

m2 <- ulam(
    alist(
        # Y model
        Y ~ bernoulli(p),
        logit(p) <- a[E] + g*X + h*Q[id],
        a[E] ~ normal(0,1),
        g ~ normal(0,1),
        h ~ uniform(0,2),
        # Q model
        vector[id]:Q ~ normal(0,1)
    ) , data=dat2 , chains=4 , cores=4 )

precis(m2,2,omit="Q")

post <- extract.samples(m2)

plot( post$h , post$g , pch=16 , col=grau(0.2) , cex=2 , ylab="effect of I" , xlab="effect of Q" )
abline(h=0,lty=2)

quantile(post$h)

############################
# d-separation plots
a <- 0.7
cols <- c( col.alpha(1,a) , col.alpha(2,a) )

# pipe

N <- 1000
X <- rnorm(N)
Z <- rbern(N,inv_logit(X))
Y <- rnorm(N,(2*Z-1))

plot( X , Y , col=cols[Z+1] , pch=16 )
abline(lm(Y[Z==1]~X[Z==1]),col=2,lwd=3)
abline(lm(Y[Z==0]~X[Z==0]),col=1,lwd=3)
abline(lm(Y~X),lwd=3,lty=3)

# fork

N <- 1000
Z <- rbern(N)
X <- rnorm(N,2*Z-1)
Y <- rnorm(N,(2*Z-1))

plot( X , Y , col=cols[Z+1] , pch=16 )
abline(lm(Y[Z==1]~X[Z==1]),col=2,lwd=3)
abline(lm(Y[Z==0]~X[Z==0]),col=1,lwd=3)
abline(lm(Y~X),lwd=3,lty=3)

# collider

N <- 1000
X <- rnorm(N)
Y <- rnorm(N)
Z <- rbern(N,inv_logit(2*X+2*Y-2))

plot( X , Y , col=cols[Z+1] , pch=16 )
abline(lm(Y[Z==1]~X[Z==1]),col=2,lwd=3)
abline(lm(Y[Z==0]~X[Z==0]),col=1,lwd=3)
abline(lm(Y~X),lwd=3,lty=3)
