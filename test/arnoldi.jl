using IterativeSolvers
import IterativeSolvers: Krylov, Arnoldi, Terminator, Shifts, ImplicitlyRestarted, Restarted

#Test of Arnoldi iterator
n=10
M=randn(n,n)#;M+=M'
K = Krylov(M, randn(n))
println(K)
for (iter, Ar) in enumerate(Arnoldi(K, Terminator(1e-9, n)))
    println("Iteration $iter:\t residual norm = ", norm(Ar.r))
    @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
end

#Test of shifts function using random Hessenberg
H=zeros(5,5)
for i=1:5, j=1:min(5,i+1); H[j,i]=randn(); end
println(round(H,3))
Shifts(H, 3)

# Test of implicitly and explicitly restarted Arnoldi iterators
n=15
M=randn(n,n)+im*randn(n,n)
v=randn(n)+im*randn(n)
K = Krylov(M, v)
for (iter, Ar) in enumerate(ImplicitlyRestarted(Arnoldi(K, Terminator(1e-9, n)), 5, 5))
    println("Iteration $iter:\t residual norm = ", norm(Ar.r))
    @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
end

for (iter, Ar) in enumerate(Restarted(Arnoldi(K, Terminator(1e-9, n)), n))
    println("Iteration $iter:\t residual norm = ", norm(Ar.r))
    iter==n && break
    @assert abs(norm(K.A*Ar.V-Ar.V*Ar.H) - norm(Ar.r)) < sqrt(eps())
end
