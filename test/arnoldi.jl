using IterativeSolvers
import IterativeSolvers: Krylov, Arnoldi, Terminator, Shifts, ImplicitlyRestarted, Restarted
using FactCheck

context("Arnoldi") do 
#Test of Arnoldi iterator
n=10
M=randn(n,n)#;M+=M'
K = Krylov(M, randn(n))
#= println(K) =#
for (iter, Ar) in enumerate(Arnoldi(K, Terminator(1e-9, n)))
    println("Iteration $iter:\t residual norm = ", norm(Ar.r))
    @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
end

#Test of shifts function using random Hessenberg
context("Shifts") do
H=zeros(5,5)
for i=1:5, j=1:min(5,i+1); H[j,i]=randn(); end
println(round(H,3))
Shifts(H, 3)
end

# Test of implicitly and explicitly restarted Arnoldi iterators
context("restarts") do
context("implicit") do
n=15
M=randn(n,n)+im*randn(n,n)
v=randn(n)+im*randn(n)
K = Krylov(M, v)
for (iter, Ar) in enumerate(ImplicitlyRestarted(Arnoldi(K, Terminator(1e-9, n)), 5, 5))
    println("Iteration $iter:\t residual norm = ", norm(Ar.r))
    if iter > 40 && iter%5==0
        @fact norm(Ar.r) --> less_than_or_equal(1e-5)
    end
    @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
end
end
context("explicit") do
n=15
M=randn(n,n)+im*randn(n,n)
v=randn(n)+im*randn(n)
K = Krylov(M, v)
for (iter, Ar) in enumerate(Restarted(Arnoldi(K, Terminator(1e-9, n)), n))
    println("Iteration $iter:\t residual norm = ", norm(Ar.r))
    iter==n && break
    @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
end
end

function assertritzclose(ef, tol=1e-8)
    for i in 1:length(ef.values)
        ritzval = ef.values[i]
        ritzvec = ef.vectors[:,i]
        compritzval = dot(ritzvec,(K.A*ritzvec))
        # Test that the eigenvectors correspond to the eigenvalues.
        @assert compritzval - ritzval <= 1e-10
    end
end

function orthogonality(ef)
    return vecnorm(ef.vectors'*ef.vectors - I)
end

context("access") do 
# Test that we can access the eigenvectors within iterations
n=10
M=randn(n,n); M+=M'
K = Krylov(M, randn(n))
for (iter, Ar) in enumerate(Arnoldi(K, Terminator(1e-9, n)))
    #= println("Iteration $iter:\t residual norm = ", norm(Ar.r)) =#
    ef = eigfact!(deepcopy(Ar))
    @assert length(ef.values) == iter
    assertritzclose(ef)
    lambda, istar = findmax(ef.values)
    evec = ef.vectors[:,istar]
    # Test that the eigenvectors are nearly orthogonal
    @assert orthogonality(ef) <= 1e-10
    @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
end

n=10
M=randn(n,n); M+=M'
K = Krylov(M, randn(n))
for (iter, Ar) in enumerate(ImplicitlyRestarted(Arnoldi(K, Terminator(1e-9, n)),5,5 ))
    #= println("Iteration $iter:\t residual norm = ", norm(Ar.r)) =#
    ef = eigfact!(deepcopy(Ar))
    #= println("Eigenvalues are $(ef.values)") =#
    #= @assert length(ef.values) <= 6 =#
    if length(ef.values) >= 1
        assertritzclose(ef)
        lambda, istar = findmax(ef.values)
        evec = ef.vectors[:,istar]
        # Test that the eigenvectors are nearly orthogonal
        @assert orthogonality(ef) <= 1e-10
        @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
    end
end


n=10
M=randn(n,n); M+=M'
K = Krylov(M, randn(n))
for (iter, Ar) in enumerate(Restarted(Arnoldi(K, Terminator(1e-9, n)),5))
    #= println("Iteration $iter:\t residual norm = ", norm(Ar.r)) =#
    ef = eigfact!(deepcopy(Ar))
    #= println("Eigenvalues are $(ef.values)") =#
    #= @assert length(ef.values) <= 6 =#
    if length(ef.values) >= 1
        assertritzclose(ef)
        lambda, istar = findmax(ef.values)
        evec = ef.vectors[:,istar]
        # Test that the eigenvectors are nearly orthogonal
        @assert orthogonality(ef) <= 1e-10
        @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
    end
    if iter == n
        break
    end
end
end
end
end
