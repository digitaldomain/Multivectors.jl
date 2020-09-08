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

"""
Extension of Blades into a vector space.  

KVectors are containers for Blades of the same grade.  New Blade elements are added with the + operator
"""

export 
KVector,
NullKVector,
sortbasis,
isnull,
coords,
prune,
normalize_safe,
basis_1vector

include("Blades.jl")
using StaticArrays
using LinearAlgebra

"Container of Blades with the same grade and element type"
struct KVector{T,K,N} <: AbstractArray{T,1}
  k::SVector{N,Blade{T,K}}
end

Base.getindex(B::BT, i::T) where {BT<:KVector, T<:Integer} = B.k[i]
Base.firstindex(k::K) where {K<:KVector} = firstindex(k.k)
Base.lastindex(k::K) where {K<:KVector} = lastindex(k.k)

const NullKVector{T,K} = KVector{T,K,0}

NullKVector{T,K}() where {T,K} = KVector{T,K,0}(SVector{0,T}())
isnull(b::B) where {B<:NullKVector} = true
isnull(b::B) where {B<:KVector} = false

Base.first(k::K) where {K<:KVector} = isempty(k) ? nothing : k.k[1]
Base.length(k::K) where {K<:KVector} = length(k.k)
Base.size(k::K) where {K<:KVector} = (length(k.k),)
Base.isempty(k::K) where {K<:KVector} = isempty(k.k)
Base.iterate(A::K) where {K<:KVector} = isempty(A) ? nothing : (A[1], 2)
Base.iterate(A::K, i::T) where {K<:KVector, T} = length(A) < i ? nothing : (A[i], i+1)
Base.eltype(::Type{K}) where {T,P,K<:KVector{T,P}} = Blade{T,P}

Base.conj(k::K) where {K<:KVector} = K(map(conj, k))

KVector{T}(v::Blade{T,K}) where {T,K} = KVector{T,K,1}(SArray{Tuple{1},Blade{T,K},1,1}(v))
KVector(v::Blade{T,K}) where {T,K} = KVector{T}(v)
Base.convert(::Type{B}, k::K) where {T,N,B<:KVector, K<:Blade{T,N}} = B(k)
KVector(kv::V) where {T, KN, K<:Blade{T,KN}, V<:AbstractVector{K}} = KVector{T,KN,length(kv)}(kv)
KVector(kv::KV) where KV<:KVector = kv

KVector(uv::C, 𝐼::Blade{KT,N}) where {T, N, KT, C<:SVector{N,T}} = KVector(uv.*(basis_1blades(𝐼)))

function KVector(uv::C) where {T, N, C<:SVector{N,T}} 
  @warn "resolving Blade types in top-level module via dual(1)"
  KVector(uv.*(basis_1blades(dual(1))))
end

KVector(uv::A) where {T<:Real, A<:Vector{T}} = KVector(SVector{length(uv)}(uv))
KVector(uv::A, 𝐼::Blade{KT,N}) where 
  {T<:Real, A<:Vector{T}, N, KT} = 
  KVector(SVector{length(uv)}(uv), 𝐼)

KVector(uv, ::Type{I}) where {I<:Blade} = KVector(uv, 1I)

vector(k::KA) where {K<:KVector, KA<:AbstractVector{K}} = SVector{3}.(coords.(k))

Base.one(::Type{KVector{T,K,N}}) where {T,K,N} = KVector(normalize(one(T) .* (basis_kblades(dual(1),K))))
Base.one(k::K) where K<:KVector = one(K)

Base.iszero(b::B) where {T,K,N,B<:KVector{T,K,0}} = true
Base.iszero(b::B) where {T,K,N,B<:KVector{T,K,N}} = iszero(sum((x->x*x).((k->k.x).(b.k))))

Base.:+(b::KA, c::KB) where {T<:Number,K, KA<:Blade{T,K}, KB<:Blade{T,K}} = 
  KVector{T,K,2}(SArray{Tuple{2},Blade{T,K},1,2}(b,c))
Base.:-(b::KA, c::KB) where {T<:Number,K, KA<:Blade{T,K}, KB<:Blade{T,K}} = 
  KVector{T,K,2}(SArray{Tuple{2},Blade{T,K},1,2}(b,-c))

function Base.:+(b::V, k::KVector{T,K,N}) where {T,K, V<:Blade{T,K},N} 
  i = findfirst(v->typeof(v)==V, k.k)
  if i == nothing
    KVector{T,K,N+1}(vcat(Blade{T,K}[b],k.k))
  else
    KVector(setindex(k.k,k.k[i]+b,i))
  end
end

Base.:+(b::B, k::V) where {T,K,B<:KVector{T,K},V<:Blade{T,K}} = k+b
Base.:+(b::KVector{T,K}, c::KVector{T,K}) where {T,K} = reduce( (bc,v)->v+bc, b.k; init=c )
Base.:-(b::B, k::V) where {T,K,B<:KVector{T,K},V<:Blade{T,K}} = -k+b
Base.:-(k::V, b::B) where {T,K,B<:KVector{T,K},V<:Blade{T,K}} = k+(-b)
Base.:-(a::B) where {T, B<:KVector{T}} = -one(T)*a
Base.:-(b::KVector{T,K}, c::KVector{T,K}) where {T,K} = b+(-c) 

"apply reverse to all Blades in this KVector"
Base.reverse(b::B) where B<:KVector = B(reverse.(b.k))

"""
    grade(a)

the grade k of a k-vector.  roughly speaking, the dimension of the subspace spanned by the k-vector
"""
grade(b::KVector{T,K}) where {T,K} = K 
grade(b::KVector{T,K}, i::Integer) where {T,K} = i==K ? b : zero(T)

"apply dual to all Blades in this KVector"
dual(b::KVector{T,K}) where {T,K} = KVector(dual.(b))
dual(b::KVector{T,K,0}) where {T,K} = b
"apply ⟂ to all Blades in this KVector"
⟂(b::KVector{T,K})  where {T,K} = KVector((⟂).(b)) 
⟂(b::KVector{T,K,0})  where {T,K} = b
"apply ! to all Blades in this KVector"
Base.:!(b::B) where {B<:KVector} = dual(b)

Base.zero(b::Type{B}) where {T,K,N,B<:KVector{T,K,N}} = NullKVector{T,K}()
Base.zero(b::KVector) = zero(typeof(b))

Base.:*(nb::NullKVector{T,K}, nb2::NullKVector{T,K}) where {T,K} = nb
Base.:*(nb::NullKVector{T,K}, b::KVector{T,K}) where {T,K} = nb
Base.:*(b::KVector{T,K}, nb::NullKVector{T,K}) where {T,K} = nb*b 
Base.:+(nb::NullKVector{T,K}, nb2::NullKVector{T,K}) where {T,K} = nb
Base.:+(nb::NullKVector{T,K}, b::KVector{T,K}) where {T,K} = b
Base.:+(b::KVector{T,K}, nb::NullKVector{T,K}) where {T,K} = nb+b 

Base.:*(s::T, b::B) where {T<:Real, B<:KVector} = B(s*b.k)
Base.:*(b::B, s::T) where {T<:Real, B<:KVector} = s*b
Base.:/(b::B, s::T) where {T<:Real, B<:KVector} = b*(one(T)/s)

pseudoscalar(k::K) where {K<:KVector} = pseudoscalar(first(k))

#==
"""
    inv(k)

Inverse of a KVector when it exists.  Left inverse.  k⋅inv(k) = 1
"""
Base.inv(k::K) where {K<:KVector} = reverse(k)/(k⋅reverse(k))
==#
 
"""
    ⋆(k)

Hodge star operator mapping k to it's Hodge dual.
"""
⋆(k::K) where {K<:KVector} = mapreduce(⋆,+,k)

"""
    sortbasis(b)

sort blades by bases indices in ascending order within the k-vector
"""
sortbasis(B::BT) where {BT<:KVector} = BT(sort(Vector(B.k); by=subspace∘typeof))
sortbasis(B::BT) where {BT<:Blade} = B
Base.:(==)(B::BT, B2::BT) where {BT<:KVector} = sortbasis(B).k == sortbasis(B2).k
Base.:(==)(B::BT, B2::BT2) where {BT<:KVector, BT2<:KVector} = false
#Base.:(==)(B::BT, B2::BT2) where {T,K,K2,BT<:KVector{T,K}, BT2<:KVector{T,K2}} = false

Base.promote_rule(::Type{B}, ::Type{K}) where {B<:KVector,K<:Blade} = KVector

prune(k::KVector, epsi = eps()) = sum(Iterators.filter(x->abs(scalar(x)) > epsi, k))
prune(b::Blade, epsi = eps()) = abs(scalar(b)) > epsi ? b : scalar(b)

"""
    ∧(a,b)

Wedge product between two k-vectors
"""
function ∧(A::KV,B::KV2) where {T,K,N,KV<:KVector{T,K,N}, K2,N2,KV2<:KVector{T,K2,N2}}
  AwB = Blade{T, K+K2}[]
  sizehint!(AwB, N*N2)
  for aᵢ in A.k
    for bⱼ in B.k
      awb = aᵢ∧bⱼ
      if !iszero(awb)
        push!(AwB, awb)
      end
    end
  end
  sum(AwB)
end

"""
    ∧(a,b)

Wedge product between blade and k-vector
"""
function ∧(a::BT,B::KV2) where {T,K,BT<:Blade{T,K}, K2,N2,KV2<:KVector{T,K2,N2}}
  AwB = Blade{T, K+K2}[]
  sizehint!(AwB, N2)
  for bⱼ in B.k
    awb = a∧bⱼ
    if !iszero(awb)
      push!(AwB, awb)
    end
  end
  sum(AwB)
end

∧(B::M,a::N) where {M<:KVector,N<:Blade} = -a∧B
∧(s::T,B::N) where {T<:Real,N<:KVector} = s*B
∧(B::N,s::T) where {T<:Real,N<:KVector} = s*B

"""
    gram(a, u)

The Gram matrix, a length(a)xlength(a) matrix where each entry is the dot product aᵢ⋅uⱼ 

Entries are sorted by acending basis indices from a and u.

Example:
for length(a) = 2, gram( a, u ) = [ a₁⋅u₁ a₁⋅u₂ ;
                                    a₂⋅u₁ a₂⋅u₂ ]
"""
function gram(a::K,u::L) where {K<:KVector, L<:KVector}
  a = sortbasis(prune(a))
  u = sortbasis(prune(u))
  n = length(a)
  if n != length(u)
    zero(eltype(eltype(a)))
  else
    [aᵢ⋅uⱼ for uⱼ in u for aᵢ in a] |> A->(reshape(A,n,n))
  end
end


import LinearAlgebra: det

"Determinant of Gram matrix formed from k-vectors a and u"
det(a::K,u::L) where {K<:KVector, L<:KVector} = det(gram(a,u))

import LinearAlgebra: norm,normalize,norm_sqr

norm_sqr(k::K) where {K<:KVector} = k⋅reverse(k)
norm(k::K) where {K<:KVector} = sqrt(norm_sqr(k))
normalize(k::K) where {K<:KVector} = k/norm(k)
normalize_safe(k::K) where {K<:KVector} = (kk = norm_sqr(k); abs(kk) > 0 ? k/norm(k) : k)

import Base: cos

"""
    cos(a, b)

cosine of angle between blades a and be spanned by the k-vectors a and b
"""
cos( a::A, b::B ) where { A<:Union{KVector, Blade}, B<:Union{KVector, Blade}} = norm(a⋅b)/(norm(a)*norm(b)) 

basis_1blades( k::K ) where {K<:KVector} = basis_1blades(first(k))

"""A KVector formed from all basis 1-vectors set to magnitude 1.0. Useful as a test direction"""
basis_1vector( k::K ) where {K<:KVector} = KVector(one.(basis_1blades(k)))


"""
    coords(k)

coordinate scalar values for ordered k-vector basis of k.
"""
function coords( k::KV ) where {T,K,KV<:KVector{T,K}}
  be = basis_kblades(first(k),K)
  kb = untype.(k)
  [ (i = findfirst(kbᵢ->beᵢ==kbᵢ, kb); isnothing(i) ? zero(T) : scalar(k[i])) for beᵢ in be ]
end

coords( b::Blade ) = coords(KVector(b))

factor( k::K ) where K<:KVector = factor.(k)

outermorphism(L, k::K) where K<:KVector = mapreduce(b->outermorphism(L, b), +, k)

Base.in(be::BK, bs::BK2) where {BK<:Union{Blade,KVector}, BK2<:Union{Blade,KVector}} = iszero(be∧bs)

