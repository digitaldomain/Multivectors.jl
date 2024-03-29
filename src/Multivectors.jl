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
⨽,
symmetricdot,
∙,
kvectors,
involute,
cayley_table,
cayley_matrix_description,
matrix_representation,
newton_inv,
shirokov_inv,
lift


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

function pseudoscalar(U::M) where {T, M<:Multivector{T}} 
  for i in 1:length(U.B) 
    if length(U.B[i]) > 0
      return pseudoscalar(first(U.B[i]))
    end
  end
  pseudoscalar(one(T)*e₁)
end

Base.zero(m::C) where {T, C <:Multivector{T}} = C(zero(T))

Base.conj(m::Multivector) = mapreduce(conj, +, m)

Base.:+(b::KVector{T,K}, c::Multivector{T,L}) where {T,K,L} = Multivector(b) + c
Base.:+(c::Multivector{T,K}, b::KVector{T,L}) where {T,K,L} = b+c

Base.:+(k::Blade{T,K}, c::Multivector{T,N}) where {T,N,K} = Multivector(KVector(k)) + c
Base.:+(c::Multivector{T,K}, v::Blade{T,L}) where {T,K,L} = v+c

Base.:+(c::Multivector{T,K}, s::T) where {T<:Number,K} = Multivector{T,K}(c.s+s,c.B)
Base.:+(s::R, c::Multivector{T,K}) where {T,R<:Real,K} = c+T(s)

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

scalar_or_ps(A) = iszero(A.s) ? A.s : A.s*pseudoscalar(A)

dual(A::M) where {M<:Multivector} = mapreduce(dual,+,[A.B...]; init = Multivector(scalar_or_ps(A)))
Base.:!(A::M) where {M<:Multivector} = dual(A)
rc(A::M) where {M<:Multivector} = mapreduce(rc,+,[A.B...]; init = Multivector(scalar_or_ps(A)))
lc(A::M) where {M<:Multivector} = mapreduce(lc,+,[A.B...]; init = Multivector(scalar_or_ps(A)))

Base.promote_rule(::Type{M}, ::Type{B}) where {B<:KVector,M<:Multivector} = Multivector
Base.promote_rule(::Type{M}, ::Type{K}) where {K<:Blade,M<:Multivector} = Multivector

Base.:*(B::BT, B2::BT2) where {T,K,K2,N,N2,BT<:KVector{T,K,N},BT2<:KVector{T,K2,0}} = Multivector{T,0}()
Base.:*(B::BT, B2::BT2) where {T,K,K2,N,N2,BT<:KVector{T,K,0},BT2<:KVector{T,K2,N2}} = Multivector{T,0}()
Base.:*(a::BT, b::BT2) where {T,K,K2,BT<:KVector{T,K,0},BT2<:KVector{T,K2,0}} = Multivector{T,0}()
Base.:*(B::BT, B2::BT2) where {T,K,K2,N,N2,BT<:KVector{T,K,N},BT2<:KVector{T,K2,N2}} = mapreduce(((k1,k2),)->B[k1]*B2[k2], +, Iterators.product(1:N, 1:N2))

Base.:*(M::MT, M2::MT2) where {T,S,N,N2,MT<:Multivector{S,N},MT2<:Multivector{T,N2}} = mapreduce(((b1,b2),)->M[b1]*M2[b2], +, Iterators.product(0:N, 0:N2))

Base.:*(A::K, b::B) where {K<:KVector,B<:Blade} = A*KVector(b)
Base.:*(b::B, A::K ) where {B<:Blade,K<:KVector} = KVector(b)*A

Base.:*(a::C, b::D) where {C<:CliffordNumber, D<:CliffordNumber} = Multivector(a)*Multivector(b)

Base.:*(M::MT, s::T) where {T<:Real, MT<:Multivector} = MT(s*M.s, (b->s*b).(M.B))
Base.:*(s::T, M::MT) where {T<:Real, MT<:Multivector} = M*s
Base.:/(M::MT, s::T) where {T<:Real, MT<:Multivector} = M*(one(T)/s)
Base.:/(A::M,B::N) where {M<:CliffordNumber,N<:CliffordNumber} = A*inv(B) 

sort_basis(s::T) where T<:Real = s

Base.:(==)(a::M,b::M) where {M<:Multivector} = (a.s==b.s) && sort_basis.(kvectors(prune(a)))==sort_basis.(kvectors(prune(b)))
Base.:(==)(a::M,b::N) where {M<:Multivector, N<:Multivector} = (a.s==b.s) && 
  sort_basis.(kvectors(prune(a)))==sort_basis.(kvectors(prune(b)))

for M in [Real, Multivector, Blade, KVector]
  for N in [Real, Multivector, Blade, KVector]
    if((M==Multivector && N!=Multivector) || (M!=Multivector && N==Multivector))
      Base.:(==)(a::M,b::N) = Multivector(a) == Multivector(b)
    end
  end
end


Base.reverse(a::M) where M<:Multivector = a.s+mapreduce(reverse, +, a.B; init=M())
Base.reverse(s::T) where T<:Real = s
Base.:~(k::K) where {K<:CliffordNumber} = reverse(k)

involute(v::CliffordNumber) = mapreduce(vᵢ->(-1)^grade(vᵢ)*vᵢ, +, v)
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

rcontraction(A,B) = A∨rc(B)
rcontraction(A::C, s::S) where {C<:CliffordNumberR, S<:Real} = A*s
rcontraction(A::C, s::S) where {C<:CliffordNumberR, T, S<:Multivector{T,0}} = A*s[0]
rcontraction(A::C, s::S) where {C<:Real, T, S<:Multivector{T,0}} = A*s[0]
rcontraction(A::C, s::S) where {U, C<:Multivector{U,0}, T, S<:Multivector{T,0}} = A[0]*s[0]
rcontraction(s::S, A::M) where {S<:Real, M<:Multivector} = sum(map(b->rcontraction(s, b), A)) 
rcontraction(s::S, A::M) where {U, S<:Multivector{U,0}, M<:Multivector} = sum(map(b->rcontraction(s, b), A)) 
const ⨽ = rcontraction

"""
  ∨(a,b)
  
Regressive product.           
Related to the ∧ product 

a∨b = (̄a̲∧̄b̲)̄

or 
right_complement(left_complement(a)∧left_complement(b))  
or
left_complement(right_complement(a)∧right_complement(b))  
"""
∨(a, b) = rc(lc(a)∧lc(b))
#!me ∨(a, b) = ⋆(⋆(a)∧⋆(b))

#⋅(a::A, b::B) where {TA,TB,A<:Blade{TA}, B<:Blade{TB}} = ⋆(⋆(b)∧a)

#!me
#==
some notes on vee, star and inner product
John Browne: rc(A∧B) == rc(A)∨rc(B) is axiomatic and needed to develop Grassmann Algebra for any metric
John Browne: A⋅B == A∨rc(B)
rc may be something more in later chapters though

<https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1668&context=facpub>
"Teaching electr eaching electromagnetic field theor omagnetic field theory using diff y using differential forms"
A⋅B = ⋆(⋆B∧A)
a = b∧(b⋅a) + b⋅(b∧a)


using John Browne we end up with 1e₁₂⋅1e₁₂ = 1, which we do not want
using the Teaching electo... differential forms we get the standard dot product.  Maybe we should use that?
TODO: define ⋅(A, B) = ⋆(⋆B∧A) and then see if ∨ and Browne works out
or maybe this already works out?  add tests and see where it breaks, if it passes, yay.
hmmm... would be sooo nice if this worked out:  Uses everything above
A⋅B == A∨⋆B == ⋆(⋆B∧A) == ⋆⋆B∨⋆A

First attempt didn't work well.  seems to break in ++++ ===> 1e₃⋅1e₃ = -1

==#

∨(s::T, b::C) where {T<:Real, C<:CliffordNumber} = zero(s)
∨(b::C, s::T) where {T<:Real, C<:CliffordNumber} = zero(s)
∨(s::M, b::C) where {T, M<:Multivector{T,0}, C<:CliffordNumber} = zero(s)
∨(b::C, s::M) where {T, M<:Multivector{T,0}, C<:CliffordNumber} = zero(s)

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

ishomogenous(v::M) where {M<:CliffordNumberR} = true 
ishomogenous(v::M) where {M<:Multivector} = sum(!iszero(b) for b ∈ v) == 1

Base.inv(v::M) where {M<:CliffordNumber} = shirokov_inv(v)

scalarscalar(m::M) where M<:Multivector = mapreduce(scalarscalar, +, m) 
scalarscalar(k::K) where K<:KVector = mapreduce((x->x*x)∘scalar, +, k)
scalarscalar(b::B) where B<:Blade = b.x*b.x
scalarscalar(s::R) where R<:Real = s*s

prune(s::Real, epsi = eps()) = abs(s) < epsi ? zero(s) : s
prune(A::M, epsi = eps()) where {T, M<:Multivector{T}} = 
  reduce(+, Iterators.filter(k->scalarscalar(k) > epsi*epsi, map(prune, A)); init=zero(T))

# WIP pass in some matrix basis representations and string names
#==
function cayley_table(b, s)
  d = length(b)
  ct = DataFrame()
  ss = vcat( vcat(s, [s[i]*s[j] for i in 1:d for j in i:d]),
            s[1]*s[2]*s[3])
  ct[:, :X] = ss
  bb = vcat(vcat(b, [b[i]*b[j] for i in 1:d for j in i:d]),
            [b[1]*b[2]*b[3]])

  ssf = vcat(["1", "-1"], ss)
  bbf = vcat([one(b[1]), -one(b[1])], bb)
  for i in 1:length(bb)
    bᵢb = bb .* [bb[i]]
    c = map(bᵢb) do bᵢbⱼ
      br = findfirst(isequal(bᵢbⱼ), bbf)
      if isnothing(br)
        "-"*ssf[findfirst(isequal(-bᵢbⱼ), bbf)]
      else
        ssf[br]
      end
    end
    ct[:, ss[i]] = c
  end

  ct
end
==#

function cayley_table(b, s; id = one(b[1]))
  d = length(b)
  ct = DataFrame()

  ss = mapreduce(c->reduce(*,c), vcat,  combinations(s))
  ct[:, :X] = ss

  bb = mapreduce(c->reduce(*,c), (a,c)->vcat(a, [c]),  combinations(b); init=[])

  ssf = vcat(["1", "-1", "Ø"], ss)
  bbf = vcat([id, -id, zero(b[1])], bb)
  for i in 1:length(bb)
    bᵢb = bb .* [bb[i]]
    c = map(bᵢb) do bᵢbⱼ
      br = findfirst(isapprox(bᵢbⱼ), bbf)
      if isnothing(br)
        br = findfirst(isapprox(-bᵢbⱼ), bbf)
        if isnothing(br)
          tr(bᵢbⱼ)
        else
          "-"*ssf[br]
        end
      else
        ssf[br]
      end
    end
    ct[:, ss[i]] = c
  end

  ct
end

cayley_tuples(b) = [(r, c, b[r]*b[c]) for r in 1:length(b) for c in 1:length(b)]
cayley_tuples(b::Vector{T}) where T<:Type = cayley_tuples(one.(b))

function cayley_table(e=dual(1))
  b = vcat(1, mapreduce( i->one.(basis_kblades(e,i)), vcat, 1:grade(pseudoscalar(e)))) 
  b*b'
end

cayley_table(E::Vector) = b*b'

function cayley_matrix_description(b::Vector)
  vcr = map(((r,c,v),)->(v,c,r), cayley_tuples(b))
  v2r = Dict()
  for (i,bᵢ) in enumerate(b)
    v2r[one(bᵢ)] = i
  end

  A = Array{Number, 2}(undef, length(b), length(b))
  A[:,:] = fill(zero(scalar(first(b))), length(b), length(b))
  for (v,c,r) in vcr
    vo = sign(v)
    if !iszero(v)
      A[v2r[abs(v)], c] = vo*b[r]
    end
  end
  
  A
end

"""
    matrix_representation(eᵢ)

build a matrix representation of the blade eᵢ using the Cayley table.
this is a larger matrix than you would get compared to a Dirac Matrix style of generation.
"""
function matrix_representation(b::Type{B}) where B<:Blade
  C = cayley_matrix_description(b)
  map(cᵢ->typeof(cᵢ) <: b ? sign(cᵢ) : zero(scalar(cᵢ)), C)
end

Base.Array(M::Multivector{T}) where T = mapreduce(b->b.x*(matrix_representation∘untype)(b), +, mapreduce(k->[b for b in k], vcat, [ k for k in M]))

cayley_matrix_description(e=dual(1)) = cayley_matrix_description(vcat(1, mapreduce( i->one.(basis_kblades(e,i)), vcat, 1:grade(dual(1))))) 

#== pixar dynamic deformables: the search for a GA basis
==#

# try to find a basis using pauli matrices with only real entries ( there are two ) that we can 
# use to build the Q eigenmatrices
# http://graphics.pixar.com/library/AnalyticEigensystems/paper.pdf
# for example:  qbase(false, 1) * qbase(true, 1) = is the x-axis twist matrix T
# whih is a flip matrix * a pinch matrix
qbase(pauli_isdiag, zerorc) = begin pauli = pauli_isdiag ? [1.0 0; 0 -1] : [0.0 1; 1 0];  Qb = zeros(3,3); Qb[setdiff([1,2,3],[zerorc]), setdiff([1,2,3],[zerorc])] = pauli; Qb; end


factor( M::MT ) where MT<:Multivector = factor.(kvectors(M))

outermorphism(L, M::MT) where MT<:Multivector = scalar(M) + mapreduce(k->outermorphism(L, k), +, kvectors(M))

Base.in(be::BK, bs::BK2) where {BK<:CliffordNumber, BK2<:CliffordNumber} = iszero(be∧bs)

Base.:^(B::T, n::Integer) where T<:CliffordNumber = prod([B for i in 1:n])

"""
    exp(M)

Taylor series approx exponential of a CliffordNumber around 0
"""
function Base.exp(M::T; n = 20, tol = eps(fieldtype(M))) where T<:CliffordNumber
  FT = fieldtype(M)
  d = 1.0
  eᴹ = one(FT)
  emf = one(FT)
  f = one(FT)

  while n > 0
    emf *= M/f
    f += one(FT)
    eᴹ′ = eᴹ + emf
    d = LinearAlgebra.norm_sqr(eᴹ′-eᴹ)
    eᴹ = eᴹ′
    n -= 1
    abs(d) > tol || break
  end

  eᴹ
end

Base.exp(B::CliffordNumber, n) = 1.0+sum([ (B^i)/prod(1:i) for i in 1:n])

"""
    log(M)

Taylor series approx natural logarithm of a CliffordNumber around 1.  
1 is choosen as expansion point for symmetry with exp at 0.
e⁰ == 1, log(e⁰) == log(1) == 0 
"""
function Base.log(M::T; n = 20, tol = eps(fieldtype(M))) where T<:CliffordNumber
  FT = fieldtype(M)
  d = 1.0
  logM = zero(FT)
  logMf = -one(FT)
  f = one(FT)
  M_1 = M - one(FT)

  while n > 0
    logMf *= -M_1
    logM′ = logM + logMf/f
    f += one(FT)
    d = LinearAlgebra.norm_sqr(logM′-logM)
    logM = logM′
    n -= 1
    d > tol || break
  end

  logM
end

fieldtype(M::MT) where {F<:Number, MT<:Multivector{F}} = F


"""
    newton_inv(m)

find reciprocal of a CliffordNumber using Newton-Rhaphson
"""
function newton_inv(m, 
                    x₀=(~m)/grade(m*(~m), 0),
                    errtol²=0.0001, 
                    maxiter=15)
  n = 0.75/(norm(m))
  m *= n
  xᵢ = x₀

  while maxiter > 0
    xᵢ′ = 2.0*xᵢ-xᵢ*m*xᵢ
    r = LinearAlgebra.norm_sqr(xᵢ′ - xᵢ)
    xᵢ = xᵢ′
    errtol² >= r && break
    maxiter -= 1
  end
  xᵢ*n
end


"""
conjugation from Shirokov inverse paper
"""
Δⱼ(v::CliffordNumber, j::Int) = mapreduce(vᵢ->vᵢ*(-1)^binomial(grade(vᵢ), 2^(j-1)), +, v)

# helpers for skirokov_inv
C(U::CliffordNumber, k, N) = grade(U, 0)*N/k

"""
  shirokov_inv(U)

  algebraic inverse of multivector of arbitrary grade in a non-degenerate algebra

  Ref: On determinant, other characteristic polynomial coefficients, and inverses in Clifford algebras of arbitrary dimension
D. S. Shirokov

"""
function shirokov_inv(U::CliffordNumber)
  n = grade(pseudoscalar(U))
  N = 2^div(n+1, 2)

  Uk = U
  local AdjU
  for k in 2:N
    AdjU = C(Uk, k-1, N) - Uk
    Uk = -U*AdjU
  end

  DetU = -grade(Uk, 0) # -Uk should only have scalar part
  AdjU/DetU
end

shirokov_inv(s::Real) = inv(s)

function Base.isapprox(M::MT, N::MT2; kwargs...) where {MT<:CliffordNumber, MT2<:CliffordNumber} 
  mapreduce((b,c)->isapprox(b,c; kwargs...), (acc,e)->acc && e, Multivector(M), Multivector(N))
end

"""
    lift(k, m)

lift a member (Multivector, KVector, or Blade) of a lower dimensional algebra k into
the even subalgebra of another algebra that includes the non-euclidean basis 1-blade m.
The bivector generated by lifting is isomorphic to the original member.

example:

    julia> module f3
             using Multivectors
             @generate_basis("++-")
           end

    julia> using .f3

    julia> @generate_basis("++") 

    julia> lift.([1e₁, 2e₂, 12e₁₂], f3.e₃)
    3-element Array{Blade{Int64,2},1}:
      1Main.f3.e₁₃
      2Main.f3.e₂₃
     12Main.f3.e₁₂

    julia> ans .* ans
    3-element Array{Int64,1}:
        1
        4
     -144

    julia> [1e₁, 2e₂, 12e₁₂] .* [1e₁, 2e₂, 12e₁₂]
    3-element Array{Int64,1}:
        1
        4
     -144
  
"""
lift(k, m::Type{T}) where T<:e₋ = sum([lift(i, m) for i in k])

lift(k::Vector, m::Type{T}) where T<:e₋ = sum([lift(i, m) for i in k])

lift(s::Real, m::Type{T}) where T<:e₋ = s

function lift(b::Blade{R, 1}, m::Type{T}) where {R, T<:e₋} 
  n = grade(pseudoscalar(m))
  scalar(b) * (alle(m,n)[subspace(b)]∧m)
end

function lift(b::Blade{R, 2}, m::Type{T}) where {R, T<:e₋}
  n = grade(pseudoscalar(m))
  scalar(b) * reduce(∧, alle(m,n)[subspace(b)])
end

lift(k, F::Module) = sum([lift(i, F) for i in k])

lift(k::Vector, F::Module) = sum([lift(i, F) for i in k])

lift(s::Real, F::Module) = s

function lift(b::Blade{T, 1}, F::Module) where T 
  n = grade(pseudoscalar(F))
  scalar(b) * (alle(F,n)[subspace(b)]∧last(basis_1blades(F)))
end

function lift(b::Blade{T, 2}, F::Module) where T
  n = grade(pseudoscalar(F))
  scalar(b) * reduce(∧, alle(F,n)[subspace(b)])
end



end # module
