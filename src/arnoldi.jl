import Base: start, next, done, eigfact!, eigfact, eigvals

#The usual Krylov space machinery
type Krylov{T}
    A
    v₀ :: AbstractVector{T}
end

#Termination criteria
type Terminator
    tol :: Real
    maxiter :: Int
end

# __Definition 2.1.__ The Arnoldi factorization is a truncated Hessenberg factorization of a matrix $A$ such that
#
# $$ AV = VH + r e_k^T $$
#
# where $V \in \mathbb{R}^{n\times k}$, $V^T V = I_k$, $H \in \mathbb{R}^{k\times k}$ is upper Hessenberg, $r \in \mathbb{R}^n$ with $0=V^T r$.

type ArnoldiFact{T} <: Factorization{T}
    V :: Matrix{T} #An orthogonal basis for the kth Krylov subspace
    H :: Matrix{T}
    r :: Vector{T}
end

function ArnoldiFact{T}(V::Matrix{T}, H::Matrix{T}, r::Vector{T}; docheck::Bool=true)
    #Check dimensions
    @assert size(V,2) == size(H,1)
    @assert size(V,1) == size(r,1)
    if docheck #Check Arnoldi projection identities
	#XXX TODO
    end
    ArnoldiFact{T}(V, H, r)
end

#The Arnoldi iterator

abstract Factorizer{T}

type Arnoldi{T} <: Factorizer{T}
    K :: Krylov{T}
    term :: Terminator
end

function start{T}(P::Arnoldi{T})
    r = P.K.v₀
    n = size(r, 1)
    V = zeros(T,n,0)
    H = zeros(T,0,0)
    ArnoldiFact{T}(V, H, r)
end

# __The `Arnoldi` function (Algorithm 3.7; Sorensen, 1992)__.
#
# Input: $AV - VH = re_k^T$ with $V^T V = I_k, V^T r = 0$
#
# Output: $AV - VH = re_{k+p}^T$ with $V^T V = I_{k+p}, V^T r = 0$

function next{T}(P::Arnoldi{T}, F::ArnoldiFact{T})
    V, H, r = F.V, F.H, F.r
    K = P.K
    β = norm(r)

    H = size(H,2)==0 ? zeros(1,0) :
      [H; [zeros(1,size(H,2)-1) β]]
    v = r / β
    V = [V v]

    #Construct next Krylov vector iterate and project into orthogonal
    #complement of existing Krylov subspace
    w = K.A*v

    h = V'w
    H = [H h]

    r = w - V*h
    #Gram-Schmidt
    #Iterative reorthogonalization of Daniel, Gragg, Kaufman and Stewart, 1976
    for i=1:2
      s = V'r
      r-= V*s #orthogonalize r
      h+= s   #update orthogonalization coefficients
      norm(s) < P.term.tol*norm(r) && break
    end
    @assert abs(norm(K.A*V-V*H) - norm(r)) < P.term.tol
    F  = ArnoldiFact(V, H, r)
    F, F
end

function done(P::Factorizer, F::ArnoldiFact)
    #TODO maxiter
    β = norm(F.r)
    β < P.term.tol
end

function Shifts(H, p, by::Function=x->real(x))
    #by: Keep eigenvalues by algebraically largest real part

    #Use exact shifts
    evals = eigvals(H)
    #Select p unwanted eigenvalues
    0≤p≤size(H,1) || throw(ArgumentError("Asked for p=$p eigenvalues but only $(size(H,1)) are available"))
    sort!(evals, by=by)
    evals[1:p]
end

# Implicitly restarted Arnoldi (Algorithm 3.8; Sorensen, 1992)
# XXX the signature should really be chained: T, Alg<:Factorizer{T}
type ImplicitlyRestarted{Alg<:Factorizer} <: Factorizer
    P :: Alg
    k :: Int #Maximum size of Arnoldi factorization
    p :: Int #Number of extra steps
    getshifts :: Function
end


function ImplicitlyRestarted{Alg<:Factorizer}(P::Alg, k::Int, p::Int, getshifts::Function=Shifts)
    ImplicitlyRestarted{Alg}(P, k, p, getshifts)
end

#Explicitly restarted
#See Saad, 1992
#Use Chebyshev polynomials?
#THis is the dumb restart where you just drop all the old vectors
type Restarted{Alg<:Factorizer} <: Factorizer
    P :: Alg
    k :: Int #Maximum size of Arnoldi factorization
end

start(P::Union(ImplicitlyRestarted,Restarted)) = start(P.P)

function next{T}(P::Restarted{Arnoldi{T}}, F::ArnoldiFact{T})
    F, _ = next(P.P, F)
    if size(F.H,1) == P.k
        n = size(F.r, 1)
        V = zeros(T,n,0)
        H = zeros(T,0,0)
        F = ArnoldiFact{T}(V, H, F.r)
    end
    F, F
end

function next{T}(P::ImplicitlyRestarted{Arnoldi{T}}, F::ArnoldiFact{T})
    F, _ = next(P.P, F)
    if size(F.H,1)==P.k+P.p #Deflate down to size k again
        k, p = P.k, P.p
        H, V, r = F.H, F.V, F.r

        u = P.getshifts(H, P.p) #also roots of the so-called filter polynomial

        Q = I
        for j=1:p
            Qj, Rj = qr(H-u[j]*I)
            H = Qj'H*Qj #TODO replace with bulge-chasing
            Q = Q*Qj
        end

        VQ = V*Q
        v=VQ[:,1+k]
        V=VQ[:,1:k]

        β=H[k+1,k]
        σ=Q[k+p,k]
        r=v*β + r*σ
        F = ArnoldiFact(V, H[1:k,1:k], r)
    end
    F, F
end

done(P::Union(ImplicitlyRestarted,Restarted), F::ArnoldiFact) = done(P.P, F)




# Get approximate eigenvalues and eigenvectors
# The usual method is to approximate them with the Ritz vectors and values
# satisfying the Galerkin condition
#
#     w⋅(Ax - xθ) = 0 ∀w ∈Krylov(A,v) of size k
#
# The Ritz values are just the eigenvalues of Ar.H
# and the Ritz vectors can be formed by lifting the eigenvectors of Ar.H
# back into the original basis (essentially undoing the projection into the
# Krylov subspace).
#
# Par80 gives additional rigorous error bounds on the eigenvalues.
#
# purge=false (default) performs an additional correction to the eigenvectors
# known as purging as suggested in Ericsson and Ruhe, 1980.
# Use purge only if the computed eigenvalues are suspected to be large
# (purge=true recommended for shift-and-invert transformed problems)
function eigfact!(Ar::ArnoldiFact; purge::Bool=false)
    fact = eigfact!(Ar.H)
    evals, evecs = fact.values, Ar.V*fact.vectors
    if purge
        for i in 1:length(evals)
            evecs[:,i] += fact.vectors[end,i]/evals[i]*Ar.r
            evecs[:,i] /= norm(evecs[:,i])
        end
    end
    Base.Eigen(evals, evecs)
end

eigfact(Ar::ArnoldiFact; purge::Bool=false) =
    eigfact!(ArnoldiFact(Ar.V, copy(Ar.H), Ar.r), purge=purge)
eigvals(Ar::ArnoldiFact) = eigvals(Ar.H)


n=10
M=randn(n,n); M+=M'
K = Krylov(M, randn(n))
for (iter, Ar) in enumerate(Arnoldi(K, Terminator(1e-9, n)))
    println("Iteration $iter:\t residual norm = ", norm(Ar.r))
    ef = eigfact!(deepcopy(Ar))
    @show ef.values
    lambda, istar = findmax(ef.values)
    evec = ef.vectors[:,istar]
    @show evec
    @assert dot(evec,(K.A*evec)) - lambda <= 10.0^-7
    @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
end

# # References
#
# 1. D. C. Sorensen, "Implicit application of polynomial filters in a $k$-step Arnoldi method", _SIAM J. Matrix Anal. Appl._ 13 (1), pp. 357-385, January 1992. [doi:10.1137/0613025](http://epubs.siam.org/doi/abs/10.1137/0613025)
