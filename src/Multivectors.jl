# Copyright 2020 Digital Domain 3.0
#
# Licensed under the Apache License, Version 2.0 (the "Apache License")
# with the following modification; you may not use this file except in
# compliance with the Apache License and the following modification to it:
# Section 6. Trademarks. is deleted and replaced with:
#
# 6. Trademarks. This License does not grant permission to use the trade
#    names, trademarks, service marks, or product names of the Licensor
#    and its affiliates, except as required to comply with Section 4(c) of
#    the License and to reproduce the content of the NOTICE file.
#
# You may obtain a copy of the Apache License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Apache License with the above modification is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the Apache License for the specific
# language governing permissions and limitations under the Apache License.

module Multivectors

export 
Multivector,
KVectorVector,
scalarprod,
grades,
∨,
lcontraction,
rcontraction,
symmetricdot,
∙,
kvectors,
involute

using Base.Iterators
using StaticArrays
using SparseArrays
using LinearAlgebra

include("KVectors.jl")

magnitude(s::Real) = s

const KVectorVector{T} = Vector{KVector{T}}

struct Multivector{T,N} <: Number
  s::T
  B::Vector{KVector{T}}  #!me change to StaticVector, should be able to optimize better
end

scalar(M::Multivector) = M.s
kvectors(M::Multivector) = M.B
kvectors(s::T) where T<:Real = [s]
kvectors(k::K) where K<:Union{KVector,Blade} = [k]

#TODO: go through and use this as base case, can delete a lot of redundant code then
const CliffordNumber = Union{Blade, KVector, Multivector}
const CliffordNumberR = Union{Blade, KVector, Multivector, Real}

#==
@generated function Multivector{T,N}() where {T,N}
  B = [KVector{T,i,0}() for i in 1:N]
  :(Multivector{$T,$N}(zero($T), $B))
end
==#


@generated function Multivector{T,N}() where {T,N}
  B = SVector{N,KVector{T,KVK,KVN} where KVK where KVN}([KVector{T,i,0}() for i in 1:N])
  :(Multivector{$T,$N}(zero($T), $B))
end

#==
function Multivector(k::KVector{T,K}) where {T,K} 
  B = KVector{T}[KVector{T,i,0}() for i in 1:K]
  B[K] = k 
  Multivector{T,K}(zero(T), B)
end
==#

@generated function Multivector(k::KVector{T,K}) where {T,K} 
  #B = @SVector [KVector{T,i,0}() for i in 1:K]
  B = SVector{K,KVector{T,KVK,KVN} where KVK where KVN}([KVector{T,i,0}() for i in 1:K])
  :( Multivector{$T,$K}(zero($T), setindex($B, k, $K)) )
end

function Multivector(s::T, a::AbstractVector{KVector{T}}) where {T<:Number}
  g = grade.(a)
  N = reduce(max,g)
  B = Vector(Multivector{T,N}().B)
  B[g] = a
  Multivector{T,N}(s,B)
end

Multivector(a::AbstractVector{KVector{T}}) where {T<:Number} = Multivector(zero(T), a)

Multivector(a::Blade{T}) where T = Multivector(KVector{T}[KVector(a)])
Multivector(s::T) where {T<:Real} = Multivector{T,0}(s, KVector{T}[])
Multivector{T, K}(s::S) where {S<:Real, T<:Real, K} = Multivector{promote_type(T,S), K}() + s

function Base.show(io::IO, M::Multivector{T,N}) where {T,N}
  tosub = Dict(0=>'₀',1=>'₁',2=>'₂',3=>'₃',4=>'₄',5=>'₅',6=>'₆',7=>'₇',8=>'₈', 9=>'₉')
  println(io, typeof(M))
  sep = "⟨"
  if !iszero(M.s) 
    print(io, sep,M.s,"⟩₀")
    sep = " + ⟨"
  end
  for b in M.B
    if !iszero(b)
      for bᵢ in b.k
        print(io, sep, bᵢ)
        sep = ","
      end
      print(io,"⟩",tosub[grade(b)])
      sep = " + ⟨"
    end
  end
end

Base.IndexStyle(::Type{<:Multivector}) = Base.IndexLinear()
Base.size(M::Multivector{T,N}) where {T,N} = N+1
Base.getindex(M::Multivector{T,N},i::Integer) where {T,N} = iszero(i) ? M.s : M.B[i]
Base.getindex(M::Multivector{T,N}, r::UnitRange{TI}) where {T,N,TI} = [M[i] for i in r]
Base.getindex(M::Multivector{T,N}, r::Union{Tuple, AbstractVector}) where {T,N} = [M[i] for i in r]
Base.firstindex(M::Multivector{T,N}) where {T,N} = 0
Base.lastindex(M::Multivector{T,N}) where {T,N} = N
StaticArrays.setindex(M::MV, v, i) where {MV<:Multivector} = 
  iszero(i) ? MV(v, M.B) : MV(M.s, setindex(M.B,v,i))

Base.first(M::Multivector{T,N}) where {T,N} = M.s

function Base.iterate( A::Multivector )
  g = grades(A)
  (A[g[1]], (2, g))
end
function Base.iterate( A::TM, i_grades::Tuple{Int, Vector} ) where {T, N, TM<:Multivector{T, N}} 
  i, grd = i_grades
  if i > length(grd)
    nothing 
  else
    (A[grd[i]], (i+1, grd))
  end
end

Base.IteratorSize(::Multivector) = Base.HasLength()
Base.length(m::Multivector) = length(grades(m))

import Base: map, mapreduce

map(f, m::Multivector) = (f(k) for k in m)
mapreduce(f, r, m::Multivector) = reduce(r, (f(k) for k in m))

Base.conj(m::Multivector) = mapreduce(conj, +, m)

Base.:+(b::KVector{T,K}, c::Multivector{T,L}) where {T,K,L} = Multivector(b) + c
Base.:+(c::Multivector{T,K}, b::KVector{T,L}) where {T,K,L} = b+c

Base.:+(k::Blade{T,K}, c::Multivector{T,N}) where {T,N,K} = Multivector(KVector(k)) + c
Base.:+(c::Multivector{T,K}, v::Blade{T,L}) where {T,K,L} = v+c

Base.:+(c::Multivector{T,K}, s::T) where {T<:Number,K} = Multivector{T,K}(c.s+s,c.B)
Base.:+(s::R, c::Multivector{T,K}) where {T,R<:Real,K} = c+s

#Base.:+(b::KVector{T,K}, c::KVector{T,L}) where {T,K,L} = Multivector{T,K}(KVector{T}[b,c])
#Base.:+(b::KVector{T,K}, c::KVector{T,L}) where {T,K,L} = Multivector{T,K}(KVector{T}[b,c])
#==
function Base.:+(b::KVector{T,K}, c::KVector{T,L}) where {T,K,L} 
  B = KVector{T}[KVector{T,i,0}() for i in 1:max(K,L)]
  B[K] = b
  B[L] = c
  Multivector{T,max(K,L)}(zero(T), B)
end
==#

@generated function Base.:+(b::KVector{T,K}, c::KVector{T,L}) where {T,K,L} 
  #B = @SVector [KVector{T,i,0}() for i in 1:max(K,L)]
  B = SVector{max(K,L),KVector{T,KVK,KVN} where KVK where KVN}([KVector{T,i,0}() for i in 1:max(K,L)])
  KL = max(K,L)
  #:(begin $B[$K] = b; $B[$L] = c; Multivector{$T,$KL}(zero($T), $B); end)
  :( Multivector{$T,$KL}(zero($T), setindex(setindex($B, b, $K), c, $L)) )
end

Base.:+(c::KVector{T,L}, s::R) where {T,R<:Real,K,L} = s + c

function Base.:+(M::Multivector{T,NM}, N::Multivector{T,NN}) where {T,NM,NN}
  if NM >= NN
    kv = copy(M.B)
    for i in 1:NN
      kv[i] = kv[i] + N.B[i]
    end
    Multivector{T,NM}(M.s+N.s, kv)
  else
    kv = copy(N.B)
    for i in 1:NM
      kv[i] = kv[i] + M.B[i]
    end
    Multivector{T,NN}(M.s+N.s, kv)
  end

  #==
  n = max(NM,NN)
  if NM == NN
    kv = KVector{T}[]
    sizehint!(kv,n)
    for i in 1:n
      push!(kv, M[i]+N[i])
    end
    Multivector{T,n}( M.s+N.s, kv )
    #Multivector{T,n}( M.s+N.s, KVector{T}[ M[i]+N[i] for i in 1:n ] )
  elseif NM < NN
    Multivector{T,n}( M.s+N.s, KVector{T}[ (M[i]+N[i] for i in 1:NM)..., N[max(NM+1,1):end]... ] )
  else
    Multivector{T,n}( M.s+N.s, KVector{T}[ (M[i]+N[i] for i in 1:NN)..., M[max(NN+1,1):end]... ] )
  end
  ==#
end

Base.:+(b::Blade{T,K}, c::Blade{T,L}) where {T<:Number,K,L} = KVector(b)+KVector(c)
Base.:+(b::B, c::K) where{B<:KVector, K<:Blade} = b+KVector(c)
Base.:+(b::K, c::B) where{B<:KVector, K<:Blade} = KVector(b)+c
Base.:+(s::T, c::B) where {T<:Real,B<:KVector} = Multivector(s)+c
Base.:+(s::T, c::K) where {T<:Real,K<:Blade} = s+KVector(c)
Base.:+(c::K, s::T) where {T<:Real,K<:Blade} = s+c

Base.:-(a::M) where {T,M<:Multivector{T}} = (-one(T))*a 
Base.:-(a::M, b::N) where {M<:CliffordNumberR, N<:CliffordNumberR} = a+(-b) 

"""
  grade(M, k)

The k-vector of grade k in multivector M
"""
function grade(M::Multivector{T,N}, k) where {T,N} 
  if( k == 0 )
    M.s
  else
    if k <= N 
      isnull(M[k]) ? zero(T) : M[k] 
    else
      zero(T)
    end
  end
end

"""
    grades(M)

Array containing grades of all non-zero k-vectors in multivector M
"""
function grades(M::Multivector{T,N}) where {T,N} 
  Bgrade = map(second, Iterators.filter(((v,i),)->!iszero(v), zip(M.B,1:length(M.B))))
  if !iszero(M.s) || isempty(Bgrade)
    vcat(0,Bgrade)
  else
    Bgrade
  end
end

grades(a::CliffordNumber) = grades(Multivector(a))
grades(s::Real) = [0]

"""
    grade(M)

The highest grade k-vector contained in M
"""
grade(A::M) where {M<:Multivector} = grades(A)[end]

grade(s::Real, i) = (i==0) ? s : zero(s)
grade(a::CliffordNumber, i) = grade(Multivector(a), i)

dual(A::M) where {M<:Multivector} = mapreduce(dual,+,[A.B...]; init = Multivector(dual(A.s)))
Base.:!(A::M) where {M<:Multivector} = dual(A)

Base.promote_rule(::Type{M}, ::Type{B}) where {B<:KVector,M<:Multivector} = Multivector
Base.promote_rule(::Type{M}, ::Type{K}) where {K<:Blade,M<:Multivector} = Multivector

Base.:*(B::BT, B2::BT2) where {T,K,K2,N,N2,BT<:KVector{T,K,N},BT2<:KVector{T,K2,0}} = Multivector{T,0}()
Base.:*(B::BT, B2::BT2) where {T,K,K2,N,N2,BT<:KVector{T,K,0},BT2<:KVector{T,K2,N2}} = Multivector{T,0}()
Base.:*(a::BT, b::BT2) where {T,K,K2,BT<:KVector{T,K,0},BT2<:KVector{T,K2,0}} = Multivector{T,0}()
Base.:*(B::BT, B2::BT2) where {T,K,K2,N,N2,BT<:KVector{T,K,N},BT2<:KVector{T,K2,N2}} = mapreduce(((k1,k2),)->B[k1]*B2[k2], +, Iterators.product(1:N, 1:N2))

Base.:*(M::MT, M2::MT2) where {T,N,N2,MT<:Multivector{T,N},MT2<:Multivector{T,N2}} = mapreduce(((b1,b2),)->M[b1]*M2[b2], +, Iterators.product(0:N, 0:N2))

Base.:*(A::K, b::B) where {K<:KVector,B<:Blade} = A*KVector(b)
Base.:*(b::B, A::K ) where {B<:Blade,K<:KVector} = KVector(b)*A

Base.:*(a::C, b::D) where {C<:CliffordNumber, D<:CliffordNumber} = Multivector(a)*Multivector(b)

Base.:*(M::MT, s::T) where {T<:Real, MT<:Multivector} = MT(s*M.s, (b->s*b).(M.B))
Base.:*(s::T, M::MT) where {T<:Real, MT<:Multivector} = M*s
Base.:/(M::MT, s::T) where {T<:Real, MT<:Multivector} = M*(one(T)/s)
Base.:/(A::M,B::N) where {M<:CliffordNumber,N<:CliffordNumber} = A*inv(B) 

sortbasis(s::T) where T<:Real = s

Base.:(==)(a::M,b::M) where {M<:Multivector} = (a.s==b.s) && sortbasis.(kvectors(prune(a)))==sortbasis.(kvectors(prune(b)))
Base.:(==)(a::M,b::N) where {M<:Multivector, N<:Multivector} = (a.s==b.s) && 
  sortbasis.(kvectors(prune(a)))==sortbasis.(kvectors(prune(b)))

for M in [Real, Multivector, Blade, KVector]
  for N in [Real, Multivector, Blade, KVector]
    if((M==Multivector && N!=Multivector) || (M!=Multivector && N==Multivector))
      Base.:(==)(a::M,b::N) = Multivector(a) == Multivector(b)
    end
  end
end


Base.reverse(a::M) where M<:Multivector = a.s+mapreduce(reverse, +, a.B; init=M())
Base.reverse(s::T) where T<:Real = s

involute(A::M) where {T, M<:Multivector{T}} = M(A.s, map(((k,b),)->(-one(T))^k*b, enumerate(kvectors(A))))

scalarprod(A::M,B::N) where {M<:Multivector,N<:Multivector} = grade(A*B,0)
scalarprod(A::M,B::N) where {M<:Multivector,N<:KVector}       = grade(A*Multivector(B),0)
scalarprod(A::M,B::N) where {M<:KVector,N<:Multivector}       = grade(Multivector(A)*B,0)
scalarprod(A::M,B::N) where {M<:KVector,N<:KVector}             = grade(Multivector(A)*Multivector(B),0)

∧(A::M, B::N) where {M<:CliffordNumber, N<:CliffordNumber} = Multivector(A)∧Multivector(B) 
∧(A::M, B::N) where {M<:Multivector, N<:Multivector} = 
  mapreduce(((b1,b2),)->A[b1]∧B[b2], +, Iterators.product(0:grade(A), 0:grade(B)))
∧(A::M, B::N) where {M<:Union{KVector, Blade}, N<:Union{KVector, Blade}} = 
  grade(A*B,grade(A)+grade(B))
∧(s::T, B::N) where {T<:Real, N<:CliffordNumber} = s*B
∧(A::M, s::T) where {M<:CliffordNumber, T<:Real} = s*B

#===
Scalars and inner product always a bit special
1. Scalars: 
For a O-blade a, we get lcontraction(a, B) = aB; 
if B has no scalar part, then rcontraction(B,a) = O. 
By contrast, the inner product (symmetricdot) is explicitly zero for any scalar argument.
==#

LinearAlgebra.:(⋅)(A::M, B::N) where {M<:CliffordNumber, N<:CliffordNumber} = 
  Multivector(A)⋅Multivector(B) 
LinearAlgebra.:(⋅)(A::M,B::N) where {M<:Multivector, N<:Multivector} = 
  mapreduce(((b1,b2),)->A[b1]⋅B[b2], +, Iterators.product(0:grade(A), 0:grade(B)))
LinearAlgebra.:(⋅)(A::M,B::N) where {T, M<:Union{KVector{T}, Blade{T}}, N<:Union{KVector, Blade}} = 
  (grade(B)>=grade(A)) ? grade(A*B,grade(B)-grade(A)) : zero(T)
LinearAlgebra.:(⋅)(a::T, B::N) where {T<:Real, N<:CliffordNumber} = a*B
LinearAlgebra.:(⋅)(A::N, b::T) where {T<:Real, N<:Multivector} = A⋅Multivector(b)
LinearAlgebra.:(⋅)(A::N, b::T) where {T<:Real, S, N<:Union{KVector{S}, Blade{S}}} = zero(b)
LinearAlgebra.:(⋅)(a::T1, b::T2) where {T1<:Real, T2<:Real} = a*b

rcontraction(A,B) = reverse(reverse(B)⋅reverse(A))

#==
rcontraction(A::M,B::N) where {M<:CliffordNumber, N<:CliffordNumber} =
  rcontraction(Multivector(A), Multivector(B))
rcontraction(A::M,B::N) where {M<:Multivector,N<:Multivector} = 
  mapreduce(((b1,b2),)->rcontraction(A[b1], B[b2]), +, Iterators.product(0:grade(A), 0:grade(B)))
rcontraction(A::M,B::N) where {T, M<:Union{KVector{T}, Blade{T}}, N<:Union{KVector, Blade}} = 
  (grade(A)>=grade(B)) ? grade(A*B,grade(A)-grade(B)) : zero(T)
rcontraction(a::T, B::N) where {T<:Real, N<:CliffordNumber} = a*grade(B,0)
rcontraction(B::N, α::T) where {N<:Multivector, T<:Real} = iszero(B.s) ? zero(α) : rcontraction(B, Multivector(α))
rcontraction(B::N, α::T) where {N<:CliffordNumber, T<:Real} = rcontraction(Multivector(B), α)
rcontraction(a::T1, b::T2) where {T1<:Real, T2<:Real} = a*b
==#
#==
LinearAlgebra.:(⋅)(A::M,B::N) where {M<:KVector,N<:KVector} = grade(A*B,grade(B)-grade(A))
LinearAlgebra.:(⋅)(A::M,B::N) where {M<:KVector,N<:Blade}   = grade(A*B,grade(B)-grade(A))
LinearAlgebra.:(⋅)(A::M,B::N) where {M<:Blade,N<:KVector}   = grade(A*B,grade(B)-grade(A))
==#

"""
  ∨(a,b)
  
Regressive product.  The dual of the wedge of the duals of two k-vectors or multivectors
"""
∨(a, b) = dual(dual(a)∧dual(b))

"""
  A×B

Commutator product between Multivectors A and B.  
A×B = ½(A*B-B*A)
"""
LinearAlgebra.:(×)(A::M, B::N) where {M<:CliffordNumber, N<:CliffordNumber} =  0.5*(A*B-B*A)

"""
    lcontraction(A, B)

The left contraction of A on B.  Generalizes the dot product from linear algebra.
The contraction A⌋B of an a-blade A onto a b-blade B is a sub-blade of B with grade b-a which is perpendicular to A and linear in both arguments.
The returned multivector, k-vector or blade will be the part of A contained in B and also orthogonal to A's projection on B.  
When one considers a scalar to be orthogonal to a 1-vector.  The standard dot product ⋅ defined on 1-vectors in fact does just this.  A scalar on a 1-vector is indeed orthogonal to any vectors projection on that vector. 
"""
lcontraction(A::M,B::N) where {M<:CliffordNumberR, N<:CliffordNumberR} = A⋅B

"""
    symmetricdot(A, B) 

The symmetric inner product.  

A∙B = ∑⟨⟨A⟩ᵢ⟨B⟩ⱼ⟩_|ⱼ₋ᵢ| for i,j ≠ 0, 
α∙B = 0, A∙α = 0 when α is a scalar
"""
function symmetricdot(A::M, B::N) where {M<:CliffordNumber, N<:CliffordNumber}
  if grade(A) > 0 && grade(B) > 0 
    mapreduce(((b1,b2),)->grade(grade(A, b1)*grade(B, b2), abs(b2-b1)), +, Iterators.product(1:grade(A), 1:grade(B)))
  else
    zero(grade(A,0))
  end
end

symmetricdot(α::T, B::N) where {T<:Real, N<:CliffordNumber} = zero(α)
symmetricdot(A::M, α::T) where {T<:Real, M<:CliffordNumber} = zero(α)

"""
    ∙(A, B) 

The symmetric inner product.  

A∙B = ∑⟨⟨A⟩ᵢ⟨B⟩ⱼ⟩_|ⱼ₋ᵢ| for i,j ≠ 0, 
α∙B = 0, A∙α = 0 when α is a scalar
"""
∙(A,B) = symmetricdot(A,B)

import LinearAlgebra: norm, normalize, norm_sqr
norm_sqr(A::M) where {M<:Multivector} = grade(A*reverse(A), 0)
norm(A::M) where {M<:Multivector} = sqrt(norm_sqr(A)) 
normalize(A::M) where {M<:Multivector} = A/norm(A)

# should we define and use norm instead of x*reverse(x)?
Base.inv(b::B) where {B<:KVector} = reverse(b)/grade((b*reverse(b)), 0)
Base.inv(v::M) where {M<:Multivector} = reverse(v)/grade((v*reverse(v)), 0)

scalarscalar(m::M) where M<:Multivector = mapreduce(scalarscalar, +, m) 
scalarscalar(k::K) where K<:KVector = mapreduce((x->x*x)∘scalar, +, k)
scalarscalar(b::B) where B<:Blade = b.x*b.x
scalarscalar(s::R) where R<:Real = s*s

prune(s::Real, epsi = eps()) = abs(s) < epsi ? zero(s) : s
prune(A::M, epsi = eps()) where {T, M<:Multivector{T}} = 
  reduce(+, Iterators.filter(k->scalarscalar(k) > epsi*epsi, map(prune, A)); init=zero(T))

end # module
