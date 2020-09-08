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
Simple k-vectors formed from the wedge product starting with orthogonal 1-vector basis.
Simple bivectors, trivectors, etc.

Types for blades are generated as a unique type for each different k-vector with acending indices
i.e. e₁, e₂, e₁₂
 
Equipped with a metric such that eᵢ⋅eᵢ = {1,-1,0} depending on if the metric is +,-, or 0

product of parallel basis
eᵢ*eⱼ == eᵢ⋅eⱼ when i=j 

product of orthogonal basis
eᵢ*eⱼ == eᵢ∧eⱼ == -eⱼ*eᵢ when i≠j 

operators:
* geometric product
∧ wedge/exterior product
-+ subraction, addition between blades of same type (span the same subspace )
!,⟂ orthogonal complement
~ reverse
⋆ Hodge star
"""

export 
@generate_basis,
e₀, e₊, e₋, e₊₋₀, e⁺, e⁻, e⁰, eₓʸ, Blade,
basis_1blades,
basis_kblades,
pseudoscalar,
∧,
quadmetric, 
reverse, 
grade,
⟂,
orthogonal_complement,
dual,
nonmetric_dual,
subspace,
lower,
raise,
alle,
alld,
show_basis,
kform,
magnitude,
scalar,
⋆,
factor,
untype,
outermorphism

using Combinatorics
using Base.Iterators
using FunctionWrappers: FunctionWrapper
using LinearAlgebra

partial = (f::Function,y...)->(z...)->f(y...,z...)
swap(f::Function) = (a,b...)->f(b...,a)
second(c) = c[2]

gauss_sum(first, last) = ((last-first+1)/2)*(first+last)
ishetro(c) = !mapreduce(ab->ab[1]==ab[2],&,zip(drop(c,1),take(c,length(c)-1)))

mirror(g) = vcat(g,reverse(g,dims=1))
prefix(g) = hcat(vcat(zeros(Int,size(g)[1]>>1), ones(Int,size(g)[1]>>1)), g)
graycodes( g = [0, 1] ) = prefix(mirror(g))
""" 
  gray codes of length n are the rows of the returned matrix 
"""
graycodes(n::Int) = n == 1 ? [0,1] : mapreduce(i->graycodes, (gc,gcf)->gcf(gc), 1:n-2; init = graycodes())

abstract type Blade{T,K} <: Number end
abstract type e₊₋₀{T} <: Blade{T,1} end
abstract type e⁺⁻⁰{T} <: Blade{T,1} end
abstract type e₊{T} <: e₊₋₀{T} end
abstract type e₋{T} <: e₊₋₀{T} end
abstract type e₀{T} <: e₊₋₀{T} end
abstract type e⁺{T} <: e⁺⁻⁰{T} end  
abstract type e⁻{T} <: e⁺⁻⁰{T} end  
abstract type e⁰{T} <: e⁺⁻⁰{T} end  
abstract type eₓʸ{T} <: Blade{T,1} end

quadmetric(::Type{E}) where {T, E<:e₊{T}} = one(T)
quadmetric(::Type{E}) where {T, E<:e₋{T}} = -one(T)
quadmetric(::Type{E}) where {T, E<:e₀{T}} = zero(T)
quadmetric(::Type{E}) where {T, E<:e⁺{T}} = one(T)
quadmetric(::Type{E}) where {T, E<:e⁻{T}} = -one(T)
quadmetric(::Type{E}) where {T, E<:e⁰{T}} = zero(T)

tosub(i) = [ '₁','₂','₃','₄','₅','₆','₇','₈','₉' ][i]
tosup(i) = [ '¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹' ][i]


export ZForm

"""
    ZForm{D}(f,df)

Differential 0-form
Represents a scalar function to be evaluated at a point.  Takes a vector returns a scalar.
df is a list of partial 
The index is that of the coordinate basis 1-form
D is the dimensionality of the coordinate basis 1-forms

Example:
julia> dist(x,y) = ZForm(()->(sqrt(x^2+y^2)), (i)->[x/sqrt(x^2+y^2), y/sqrt(x^2+y^2)][i])

julia> dist(1.0,-1.0).df(2)
-0.7071067811865475

julia> dist(1.0,-1.0).df(1)
0.7071067811865475

julia> dist(1.0,-1.0).f()
1.4142135623730951
"""
struct ZForm{D} <: Number  # calling a form a number is a stretch, but it works nicely inside a blade
  f # ::Function
  gradient # ::Vector{ZForm}
  hessian # ::Array{ZForm,2}
end

const KForm{D} = Blade{ZForm{D}}

"AlwaysZero(): object that always returns zero when indexed"
struct AlwaysZero
end

import Base.getindex

Base.length(x::AT) where {AT<:AlwaysZero} = 0
getindex(x::AT, i...) where {AT<:AlwaysZero} = 0
isazero(a) = false
isazero(a::T) where T<:Number = iszero(a)
isazero(b::B) where {T, B<:Blade{T}} = isazero(b.x)
isazero(z::Z) where {D,Z<:ZForm{D}} = isazero(z.f)
isazero(a::A) where {A<:AlwaysZero} = true
Base.similar(a::A) where {A<:AlwaysZero} = A

constantfun(x) = (y...)->x
zerofun = (y,z...)->zero(y)
ZForm{D}() where D = ZForm{D}((y,z...)->one(y), AlwaysZero(), AlwaysZero())
ZForm{D}(x::F) where {D,F<:Function} = ZForm{D}(x, AlwaysZero(), AlwaysZero())
ZForm{D}(f::F, df) where {D,F<:Function} = ZForm{D}(f, df, AlwaysZero())
ZForm{D}(x::T) where {D,T<:Real} = x

Base.length(zf::ZF) where {D,ZF<:ZForm{D}} = D

Base.zero(z::ZForm{D}) where D = 0.0
Base.zero(z::Type{ZForm{D}}) where D = 0.0
Base.iszero(z::ZForm{D}) where D = typeof(z.f) <: AlwaysZero 
Base.iszero(b::B) where {D, F<:ZForm{D}, B<:Blade{F}} = typeof(b.x.f) <: AlwaysZero
Base.abs(z::K) where {D, K<:ZForm{D}} = z 

mapij(f, A::AT) where {AT<:AlwaysZero} = A
mapij(f, A::AT, B) where {AT<:AlwaysZero} = A

function mapij(f, A)
  Af = similar(A)
  for ij in CartesianIndices(A)
    Af[ij] = f(Tuple(ij), A[ij])
  end
  Af
end

function mapij(f, A, B)
  ABf = similar(A)
  for ij in CartesianIndices(A)
    ABf[ij] = f(Tuple(ij), A[ij], B[ij])
  end
  ABf
end

# with second derivatives
function Base.:*(a::ZA, b::ZB) where {D,ZA<:ZForm{D}, ZB<:ZForm}
  ZForm{D}((k...)->a.f(k...)*b.f(k...), 
        map(i->(k...)->a.gradient[i](k...)*b.f(k...) + 
                       a.f(k...)*b.gradient[i](k...), 
            1:D),
        mapij((ij, addfᵢⱼ, bddfᵢⱼ)->(k...)->
                        addfᵢⱼ(k...)*b.f(k...) + 
                        a.gradient[ij[1]](k...)*b.gradient[ij[2]](k...) + 
                        a.gradient[ij[2]](k...)*b.gradient[ij[1]](k...) + 
                        a.f(k...)*bddfᵢⱼ(k...),
            a.hessian, b.hessian))
end

function Base.:*(s::R, b::ZB) where {R<:Real, D, ZB<:ZForm{D}}
  if iszero(s)
    s
  else
    ZForm{D}((k...)->s*b.f(k...), 
          isazero(b.gradient) ? b.gradient : 
                         map(dfᵢ->(k...)->s*dfᵢ(k...), b.gradient),
          isazero(b.hessian) ? b.hessian : 
                         map(bddfᵢⱼ->(k...)->s*bddfᵢⱼ(k...), b.hessian))
  end
end

Base.:*(a::ZA, s::R) where {ZA<:ZForm, R<:Real} = s*a

function Base.:+(a::ZA, b::ZB) where {D, ZA<:ZForm{D}, ZB<:ZForm{D}}
  ZForm{D}((u...)->a.f(u...) + b.f(u...), 
           map(i->(u...)->a.gradient[i](u...)+b.gradient[i](u...), 1:D),
           mapij((ij, addfᵢⱼ, bddfᵢⱼ)->(u...)->addfᵢⱼ(u...)+bddfᵢⱼ(u...), a.hessian, b.hessian))
end

Base.:+(a::ZA, s::R) where {D, ZA<:ZForm{D}, R<:Real} = iszero(s) ? a : ZA((u...)->a.f(u...)+s, a.gradient, a.hessian)

Base.:*(a::ZA, b::B) where {D, ZA<:ZForm{D}, B<:Blade} = untype(b)(b.x*a)
#!me Base.:*(a::BZ, b::B) where {Z<:ZForm{D}, BZ<:Blade{Z}, T<:Real, B<:Blade{T}} = (raise(untype(a)(one(T)))*b)
∧(a::ZA, b::B) where {D, ZA<:ZForm{D}, B<:Blade} = a*b
∧(b::B, a::ZA) where {D, B<:Blade, ZA<:ZForm{D}} = a*b
∧(a::ZA, b::ZB) where {D, ZA<:ZForm{D}, ZB<:ZForm{D}} = a*b
∧(s::R, a::ZA) where {R<:Real, D, ZA<:ZForm{D}} = s*a
∧(a::ZA, s::R) where {D, ZA<:ZForm{D}, R<:Real} = s*a
∧(a::R, b::S) where {R<:Real, S<:Real} = a*b
⋆(a::B) where {D, ZA<:ZForm{D}, B<:Blade{ZA}} = a.x*(⋆(one(untype(a))))
Base.:*(a::B, s::R) where {D, ZA<:ZForm{D}, B<:Blade{ZA}, R<:Real} = B(a.x*s) 
Base.:*(s::R, a::B) where {D, ZA<:ZForm{D}, R<:Real, B<:Blade{ZA}} = B(s*a.x)
Base.sign(a::K) where{D, K<:ZForm{D}} = 1 

Base.zero(::Type{B}) where {T,B<:Blade{T}} = zero(T)


"""
    pseudoscalar(A)

The pseudoscalar associated with algebra of blade A.  
The pseudoscalar is the singleton blade that spans the entire space.  
The highest grade blade in the algebra.
"""
pseudoscalar(a::K) where {T<:Number,K<:Blade{T}} = pseudoscalar(K)(one(T))

"""
    dual(k)

Map from the subspace k to the complementary dual subspace n-k where n is the highest grade of the algebra.
"""
#dual(E::Type{KT}) where {T<:Real, K, KT<:Blade{T,K}} = pseudoscalar(KT)
dual(R::Type{RT}) where {RT<:Real} = typeof(dual(one(R)))

"""
    ⟂(b)

The orthogonal complement of this blade against the pseudoscalar I.

defined as k*I⁻¹

If the metric is degenerate, the non-metric duality mapping dual(b) should be used instead.
"""
⟂(k::K) where {T<:Number, K<:Blade{T}} = k*inv(pseudoscalar(k))
orthogonal_complement(k) = ⟂(k)

function nonmetric_dual(k::K) where {T<:Number, K<:Blade{T}} 
  𝐼 = raise(pseudoscalar(k))
  lower(k*inv(𝐼))
end

Blade{T,0}(s::T) where T = s
Base.:+(b::KV, c::KV) where {T<:Number, K, KV<:Blade{T,K}} = KV(b.x+c.x)
Base.:-(b::KV, c::KV) where {T<:Number, K, KV<:Blade{T,K}} = KV(b.x-c.x)

"The magnitude of a blade.  Oriented."
magnitude(a::B) where {B<:Blade} = a.x
const scalar = magnitude

Base.show(io::IO, k::K) where {K<:Blade} = print(io,k.x,split(string(K), "{")[1])

strnot0(z,s) = isazero(z) ? "" : s

function Base.show(io::IO, z::Z) where {N, Z<:ZForm{N}}
  subs = ["ₓ", "y", "z", "w", "ᵢ", "ᵢ", "ᵢ", "ᵢ", "ᵢ", "ᵢ", "ᵢ", "ᵢ"]

  gstring = mapreduce(i->strnot0(z.gradient[i],", ϕ′"*subs[i]), *, 1:length(z.gradient); init = "")
  hstring = mapreduce(i->strnot0(z.hessian[i],", ϕ′′"*subs[i]), *, 1:length(z.hessian); init = "")

  print("ZForm["*strnot0(z.f,"ϕ")*gstring*hstring*"]")
end

struct BasisVectorInfo
  name::Symbol
  dualname::Symbol
  eᵢ::Vector{Int}
  eemetric::Float64 # maybe should change to Int, so as not to force promotion
  edemetric::Float64 # maybe should change to Int, so as not to force promotion
  n₋::Int64
end

function extract_basis(metric::String)
  swaps(n) = n*(n-1)/2 
  countm(s, m::Char) = (length∘findall)(c->c==m, mapreduce( i->metric[i], *, s ))

  subspacei = combinations( 1:length(metric) )
  names = map( c->mapreduce( i->tosub(i), *, c), subspacei )
  dualnames = map( c->mapreduce( i->tosup(i), *, c), subspacei )
  name = map( s->Symbol("e"*reduce(*,s)), names )
  dualname = map( s->Symbol("e"*reduce(*,s)), dualnames )
  eemetric = map( s->((-1)^swaps(length(s)))*
                      ((-1)^countm(s, '-'))*
                      (0^countm(s, '0')), subspacei )
  edemetric = map( s->((-1)^swaps(length(s)))*
                      ((-1)^countm(s, '-'))*
                      (1^countm(s, '0')), subspacei )
  n₋ = map( s->countm(s,'-'), subspacei)
  map(zip(subspacei, name, dualname, eemetric, edemetric, n₋)) do (s, n, d, m, md,nneg)
    BasisVectorInfo(n,d,s,m,md,nneg)
  end
end

"""
    alle(m,s)

An Array with all k-blade types for a given metric string s
"""
alle(gamodule, metrics) = map(bi->getproperty(gamodule,bi.name),extract_basis(metrics))

"""
    alld(m,s)

An Array with all covector types for a given metric string s
"""
alld(gamodule, metrics) = map(bi->getproperty(gamodule,bi.dualname),extract_basis(metrics))

""" 
    alle(modulename, n) 

Array with all k-blade types where k <= n"
"""
alle(gamodule, n::T) where {T<:Integer} = alle(gamodule, repeat("+",n))

""" 
    alld(modulename, n) 

Array with all dual covector types where k <= n"
"""
alld(gamodule, n::T) where {T<:Integer} = alld(gamodule, repeat("+",n))

"convenience function to show graded basis at top-level module scope"
show_basis() = alle(Main,grade(dual(1))) |> show

"implementation specific helper.  since Symbol('Main.foo') != :Main.foo need to do some name mangling"
tosym(ns,vare) = Symbol(replace(string(ns)*"_"*string(vare), "."=>"_"))

function basis_expr(name, super)
  Expr(:struct, false, Expr(:<:, Expr(:curly, name, :T), super),
      Expr(:block,
           Expr(:(::), :x, :T)))
end

function quadmetric_expr(ns, namea, nameb, abmetric)
    Expr(:(=), 
         Expr(:call, 
              Expr(:., :Multivectors, Expr(:quote, #QuoteNode 
                                     :quadmetric, 
                                    )), 
              Expr(:(::), Expr(:curly, :Val, Expr(:quote, #QuoteNode
                                                  tosym(ns,namea)))), 
              Expr(:(::), Expr(:curly, :Val, Expr(:quote, #QuoteNode
                                                  tosym(ns,nameb))))
             ), 
         Expr(:block, abmetric ))
end


function dual_expr( ns, ename, dual_fcn )
  Expr(:(=), 
       Expr(:where, 
            Expr(:call, 
                 Expr(:., :Multivectors, Expr(:quote, #QuoteNode
                                        :dual
                                       )), 
                 Expr(:(::), :k, :T)), 
            Expr(:<:, :T, ename)), Expr(:block,
                                         Expr(:call, dual_fcn, :k)
                                        ))
end

"""
    @generate_basis(s, [export_types, generate_reciprocals, zero_mixed_index])

generate types for basis Blades and algebra given a string encoding the desired metric.
example metric strings : 
"+++" for 3D space
"-+++" for minkowski spacetime
"0+++" 3D projective space ( homegeneous coords ) 

Optional boolean parameters:
export_types -       export the basis types into calling module scope

generate_reciprocals - generate reciprocal basis types with raised indices for all grades

zero_mixed_index -   any operator that results in a blade with mixed dual and primal indices will be set to zero.  You want this true when working with differential k-forms where eᵢ*eʲ = δᵢʲ
"""
macro generate_basis(metric, export_types=false, generate_reciprocals=false, zero_mixed_index=false)
  ns = __module__
  degenerate_metric = false
  e = extract_basis(metric)
  dualbasis = Vector{Expr}()
  types = Vector{Symbol}()
  basis = map(e) do eᵢ
    if length(eᵢ.eᵢ) == 1
      if isone(eᵢ.eemetric)
        et = :e₊
      elseif isone(-eᵢ.eemetric)
        et = :e₋
      else
        et = :e₀
        degenerate_metric = true
        if zero_mixed_index
          @warn "degenerate metric '0' and zeroing mixed index basis can cause dual to stay degenerate"
        end
      end
      super = Expr(:curly, et, :T)
    else
      super = Expr(:curly, :Blade, :T, length(eᵢ.eᵢ))
    end
    push!(dualbasis, 
          basis_expr(eᵢ.dualname, Expr(:curly, :Blade, :T, length(eᵢ.eᵢ)))) #Expr(:curly, :e⁺, :T)))
    push!(types, eᵢ.dualname)
    ne = length(eᵢ.eᵢ)
    if generate_reciprocals && ne > 1
      # use graycodes to get all combinations of sub/super-script
      gc = Iterators.filter(ishetro, eachrow(graycodes(ne)))
      mixede = map(gc) do (gcᵢ)
        choosescript(j) = gcᵢ[j]==1 ? tosub(eᵢ.eᵢ[j]) : tosup(eᵢ.eᵢ[j]);
        mixedname = mapreduce(j->choosescript(j), *, 1:ne; init = "e") 
        push!(types, Symbol(mixedname))
        basis_expr(Symbol(mixedname), Expr(:curly, :eₓʸ, :T))
      end
      dualbasis = vcat(dualbasis,mixede)
    end
    push!(types, eᵢ.name)
    basis_expr(eᵢ.name, super)
  end

  exporteᵢ = Vector{Expr}()
  if export_types
    exporteᵢ = map(types) do typeᵢ 
      Expr(:export, typeᵢ)
    end
  end

  ek1 = filter(partial((==), 1)∘length∘partial(swap(getproperty),:eᵢ),e) 
  quadmetric_fcn = map(ek1) do eᵢ
    quadmetric_expr(ns, eᵢ.name, eᵢ.name, eᵢ.eemetric)
  end

  if generate_reciprocals || degenerate_metric
    quadmetric_fcn = 
      vcat(quadmetric_fcn, map(ek1) do eᵢ
            quadmetric_expr(ns, eᵢ.name, eᵢ.dualname, eᵢ.edemetric)
           end)
    quadmetric_fcn = 
      vcat(quadmetric_fcn, map(ek1) do eᵢ
            quadmetric_expr(ns, eᵢ.dualname, eᵢ.name, eᵢ.edemetric)
           end)
    quadmetric_fcn = 
      vcat(quadmetric_fcn, map(ek1) do eᵢ
            quadmetric_expr(ns, eᵢ.dualname, eᵢ.dualname, eᵢ.edemetric)
           end)
  end

  if zero_mixed_index
    push!(quadmetric_fcn, quadmetric_expr( ns, :eₓʸ, :eₓʸ, 0.0 )) 
  else
    push!(quadmetric_fcn, quadmetric_expr( ns, :eₓʸ, :eₓʸ, 1.0 )) 
  end
    
  tehdual = :(⟂)
  if degenerate_metric
    tehdual = :nonmetric_dual
  end
  dual_fcn = map(types) do bᵢ
    dual_expr(ns, bᵢ, tehdual)
  end
              
  
  n = length(e[end].eᵢ)
  I₀ = e[end].name
  I⁰ = e[end].dualname
  invsign = (-1)^(n*(n-1)/2 + e[end].n₋) < 0 ? -1 : 1

  # dual of the pseudoscalar
  #==
  push!(dual_fcn,
        Expr(:(=), 
             Expr(:call, Expr(:., :Multivectors, Expr(:quote, #QuoteNode
                                                      :dual
                                                     )), Expr(:(::), :k, I₀)), 
              Expr(:block,
                        Expr(:., :k, 
                             Expr(:quote, #QuoteNode
                                  :x)))))
                                  ==#

  # dual of scalar.
  push!(dual_fcn, 
        Expr(:(=), Expr(:where, 
                        Expr(:call, Expr(:., :Multivectors, Expr(:quote, #QuoteNode
                                                                 :dual
                                                                )), 
                             Expr(:(::), :s, :T)), 
                        Expr(:<:, :T, :Real)), 
             Expr(:block,
                  Expr(:call, I₀, Expr(:call, :*, invsign, :s)))))

  # need to tie pseudoscalar to each blade type in the module.  dual(1.0) may end up in wrong etype system
  # actually what if you take the dual of an operation that returns a scalar?  could get wrong dual fcn
  # really need to have a type e₀ for scalars.  we depend on type choosing correct module.
  #!me add as a known bug
  pseudoscalar_fcn = map(e) do eᵢ
    eᵢname = eᵢ.name
    [:(Multivectors.pseudoscalar(::Type{K}) where {K<:$eᵢname} = $I₀),
     :(Multivectors.pseudoscalar(::Type{K}) where {T,K<:$eᵢname{T}} = $I₀{T})]
  end |> flatten
  # also pseudoscalar for dual space
  dpseudoscalar_fcn = map(e) do eᵢ
    eᵢname = eᵢ.dualname
    [:(Multivectors.pseudoscalar(::Type{K}) where {K<:$eᵢname} = $I⁰),
     :(Multivectors.pseudoscalar(::Type{K}) where {T,K<:$eᵢname{T}} = $I⁰{T})]
  end |> flatten
  # and for ZForm
  #!me only works for types we create via macro ( unique within module ).
  # zpseudoscalar_fcn = :(Multivectors.pseudoscalar(::Type{K}) where {K<:ZForm} = $I₀)

  return Expr(:escape, Expr(:block, 
              exporteᵢ..., 
              basis..., 
              dualbasis...,
              quadmetric_fcn...,
              pseudoscalar_fcn..., dpseudoscalar_fcn..., # zpseudoscalar_fcn,
              dual_fcn...,
             ))
end

#!me convert to generated, see the Blade{T,1} version below 
"Extract indices for basis vectors spaning the subspace of the given Blade"
function subspace( b::Type{K} ) where {K<:Blade}
  # we could probably encode all the basis information in the Blade type
  # then we wouldn't need this string based malarky.  
  # i.e. Blade{T,Tuple{Ea,Eb,Ec}} where Ea,Eb and Ec are the eᵢ indices
  fromsub = Dict('₀'=>0, '₁'=>1, '₂'=>2, '₃'=>3, '₄'=>4, '₅'=>5, '₆'=>6, '₇'=>7, '₈'=>8, '₉'=>9,
                 '⁰'=>0, '¹'=>1, '²'=>2, '³'=>3, '⁴'=>4, '⁵'=>5, '⁶'=>6, '⁷'=>7, '⁸'=>8, '⁹'=>9)
  s = split(split(string(b), ".")[end],"{")[1]
  map(sᵢ->fromsub[sᵢ], collect(s)[2:end])
end

"""
    isdualsubspace(b)

array where value is 1 if the corresponding subspace basis vector is from dualspace of the given Blade
length of array is k where k is the grade of the Blade
"""
function isdualsubspace( b::Type{K} ) where {K<:Blade}
  isdual = Dict('₀'=>0, '₁'=>0, '₂'=>0, '₃'=>0, '₄'=>0, '₅'=>0, '₆'=>0, '₇'=>0, '₈'=>0, '₉'=>0,
                '⁰'=>1, '¹'=>1, '²'=>1, '³'=>1, '⁴'=>1, '⁵'=>1, '⁶'=>1, '⁷'=>1, '⁸'=>1, '⁹'=>1)

  s = split(split(string(b), ".")[end],"{")[1]
  map(sᵢ->isdual[sᵢ], collect(s)[2:end])
end

subspace( b::K ) where {K<:Blade} = subspace(K)

#!me why this way?  inconsitent to return array for grades > 1 and atom for grade 1
@generated function subspace( b::K ) where {T,K<:Blade{T,1}}
  si = subspace(b)[1]
  :($si)
end

Base.:*(s::T, a::K) where {T<:Real, KT<:Real, K<:Blade{KT}} = K(s*a.x)
Base.:*(a::K, s::T) where {T<:Real, KT<:Real, K<:Blade{KT}} = K(s*a.x)
Base.:/(a::T, s::Real) where {R,T<:Blade{R}} = R(one(R)/s)*a
Base.:/(s::Real, a::T) where {R,T<:Blade{R}} = s*inv(a)
Base.:/(a::Blade, b::Blade) = a*inv(b)
Base.:-(a::T) where {R,T<:Blade{R}} = T(-a.x)
Base.sign(a::K) where{K<:Blade} = sign(a.x)

function swap_parity(eab)
  m = 1
  # trusty old bubble sort.
  bubn = length(eab)-1
  swapped = true
  while swapped
    swapped = false
    for i in 1:bubn
      if eab[i] > eab[i+1]
        temp = eab[i]
        eab[i] = eab[i+1]
        eab[i+1] = temp
        m = -m
        swapped = true
      end
    end
  end
  (eab,m)
end

@generated function Base.:*(a::T, b::U) where {S<:Number,T<:Blade{S}, R<:Number,U<:Blade{R}}
  ns = parentmodule(a)
  @assert ns == parentmodule(b)

  ea = zip(subspace(a), isdualsubspace(a)) |> collect
  eb = zip(subspace(b), isdualsubspace(b)) |> collect
  toscript(eᵢ) = (second(eᵢ) == 0 ? tosub(first(eᵢ)) : tosup(first(eᵢ)))
  tosyme(eᵢ) = tosym(ns,"e"*toscript(eᵢ))

  # effectively implement the geometric algebra's "multiplication table" at compile time
  allzero = false
  m = one(S)
  nea = length(ea)
  eab = Vector{Tuple{Int,Int}}()
  for i in 1:nea 
    eaᵢ = ea[i]
    j = findfirst(x->first(x)==first(eaᵢ),eb)
    if j == nothing
      push!(eab,eaᵢ)
    else
      aswaps = nea-i 
      totalswaps = ((j-1)+aswaps)
      
      m = m*(-one(S))^totalswaps
      m = m*(FunctionWrapper{S, 
                             Tuple{Val{tosyme(eaᵢ)}, 
                                   Val{tosyme(eb[j])}}}(quadmetric))(Val(tosyme(eaᵢ)), 
                                                                     Val(tosyme(eb[j])))
      splice!(eb, j)
    end
  end

  eab = vcat(eab,eb)
  eab,swp = swap_parity(eab)
  m = m*swp

  hasmixedindex(es) = begin l = mapreduce(second,+,es); l < length(es) && l > 0 end
  if length(eab) > 1 && hasmixedindex(eab)
    m = m*(FunctionWrapper{S, 
                           Tuple{Val{tosym(ns, :eₓʸ)}, 
                                 Val{tosym(ns, :eₓʸ)}}}(quadmetric))(Val(tosym(ns, :eₓʸ)), 
                                                                     Val(tosym(ns, :eₓʸ)))
  end

  if isempty(eab)
    :(a.x*b.x*$m)
  else
    if iszero(m)
      m
    else
      TU = Symbol("e"*mapreduce(toscript, *, eab))
      :($ns.$TU(a.x*b.x*$m))
    end
  end
end


"""convenience operator to build higher grade Blade Types.  indices must be in acending order"""
function Base.:*(::Type{A}, ::Type{B}) where {A<:Blade, B<:Blade}
  subs = vcat(subspace(A), subspace(B))
  ascend = mapreduce(((a,b),)->a<b, (a,b)->a&&b, zip(subs[1:end-1],subs[2:end]); init = true)
  if ascend
    untype(1A*1B)
  else
    Nothing
  end
end

"""convenience operator to build higher grade Blade Types.  indices must be in acending order"""
∧(::Type{A}, ::Type{B}) where {A<:Blade, B<:Blade} = A*B

"""
    reverse(b)

Blade constructed with reversed order of basis vectors.  
i.e. reverse(e₁∧e₂∧e₃) = e₃∧e₂∧e₁
"""
@generated function Base.reverse( a::T ) where {S<:Number,K,T<:Blade{S,K}}
  s = S((-1)^(K*(K-1)/2))
  # convert each 1-blade basis to it's dual (n-1)-blade basis
  # multiply them together to get the dual for this k-blade
  :($s*a)
end

"""
    lower(k)

direct map taking all upper indices ( dual basis ) to lower indices ( standard basis )
e.g. lower(1e¹²₃) == 1e₁₂₃
"""
@generated function lower( k::Type{K} ) where {K<:Blade}
  ns = parentmodule(K)
  TU = Symbol("e"*mapreduce(tosub, *, subspace(K)))
  :($ns.$TU)
end

lower( k::K ) where {K<:Blade} = lower(K)(k.x)
lower( s::T ) where T<:Real = s

"""
    raise(k)

direct map taking all lower indices ( standard basis ) to upper indices ( dual basis )
e.g. raise(1e¹²₃) == 1e¹²³
assumes you have generated a dual basis, i.e. @generate_basis("...", _, true)
"""
@generated function raise( k::Type{K} ) where {K<:Blade}
  ns = parentmodule(K)
  TU = Symbol("e"*mapreduce(tosup, *, subspace(K)))
  :($ns.$TU)
end

raise( k::K ) where {K<:Blade} = raise(K)(k.x)
raise( s::T ) where T<:Real = s

"""
    grade(b)

The dimension of the subspace spanned by the blade
"""
grade(s::T) where {T<:Number} = 0
grade(a::T) where {S<:Number, K, T<:Blade{S,K}} = K
grade(::Type{B}) where {B<:Blade} = grade(B{Int64})
grade(::Type{B}) where {S, K, B<:Blade{S,K}} = K

Base.:*(s::T, e::Type{K}) where {T<:Real, K<:Blade} = K(s)

"""
    inv(k)

Inverse of a blade when it exists.  Left inverse.  k*inv(k) = 1
"""
Base.inv(k::K) where {K<:Blade} = reverse(k)/(k*reverse(k))

Base.:~(k::K) where {K<:Blade} = reverse(k)
Base.:!(k::K) where {K<:Blade} = dual(k)

"""
    ∧(a,b)

Wedge product between two blades
"""
@generated function ∧(a::A, b::B) where {TA,TB,A<:Blade{TA}, B<:Blade{TB}} 
  sa = subspace(a)
  sb = subspace(b)
  if isempty(sa ∩ sb)
    :(a*b)
  else
    zero(promote_type(TA,TB)) 
  end
end

∧(s::T,b::B) where {T<:Real, B<:Blade} = s*b
∧(a::B,s::T) where {B<:Blade, T<:Real} = a*s

∧(a::T,::Type{B}) where {T<:Number, B<:Blade} = B(a)

import LinearAlgebra: ⋅
@generated function ⋅(a::A, b::B) where {TA,TB,A<:Blade{TA}, B<:Blade{TB}} 
  sa = subspace(a)
  sb = subspace(b)
  if (sa ∩ sb) == sa
    :(a*b)
  else
    zero(promote_type(TA,TB)) 
  end
end

⋅(s::T,b::B) where {T<:Real, B<:Blade} = s*b
⋅(a::B,s::T) where {TB,B<:Blade{TB}, T<:Real} = zero(promote_type(T,TB))

Base.:(==)(a::B, b::B) where {B<:Blade} = a.x == b.x
Base.:(==)(a::A, b::B) where {A<:Blade, B<:Blade} = iszero(a.x) && iszero(b.x) 
Base.:(==)(a::A, s::T) where {A<:Blade, T<:Real} = iszero(a.x) && iszero(s) 
Base.:(==)(s::T, a::A) where {A<:Blade, T<:Real} = a==s 
Base.abs(b::B) where {B<:Blade} = B(abs(b.x))

Base.isless(a::B, b::B) where B<:Blade = scalar(a) < scalar(b)

"""
    conj(b)

The complex conjugate of the blade b.
"""
Base.conj(b::B) where {B<:Blade} = reverse(b)

"""
    kform(α, u)

Apply 1-form α to u.
"""
kform( α::K, u::V ) where { TF,T,N, K<:Blade{TF,N}, V<:Blade{T,N} } = α⋅reverse(u)

import LinearAlgebra: det

det( a::B ) where { B<:Blade } = (a⋅a)/magnitude(a)

"""
  det( a, b )

  
The metric scalar product between two Blades of same grade.  
As defined in Geometric Algebra for Computer Science.
"""
@generated function det( a::A, b::B ) where { T, N, A<:Blade{T,N}, B<:Blade{T,N} }
  ea = basis_kblades(a,1)
  eb = basis_kblades(b,1)

  sa = subspace(a)
  sb = subspace(b)
  #==
  aᵢ = one(T).*ea[sa]
  bⱼ = one(T).*eb[sb]
  gramdet = det(aᵢ .⋅ reverse(bⱼ)')
  scalar(a)*scalar(b)*gramdet
  ==#
  aᵢ = ea[sa]
  bⱼ = eb[sb]
  :(a.x*b.x*det(one($T).*$aᵢ .⋅ reverse(one($T).*$bⱼ)'))
end

det( a::A, b::B ) where {T, N, M, A<:Blade{T,N}, B<:Blade{T,M}} = zero(T)

"""
    ⋆(k, 𝑖)

Hodge star operator mapping k to it's Hodge dual relative to a given psuedovector 𝑖.
Defined by k∧⋆(k) == (k⋅k)*𝑖 where k is generated from orthonormal 1-vectors.
𝑖 is the unit psuedoscalar for a subspace containing k.
"""
@generated function ⋆(b::B, i::BI) where {T, B<:Blade{T}, BI<:Blade}
  sb = subspace(b)
  n = reduce(max,sb)
  # shuffle all missing generating 1-blades from hodge dual into k to fill holes
  # track sign flips
  holes = setdiff(1:n,sb)
  # 'filling holes' is so we can make k∧⋆(k) == 𝐼. i.e. ⋆k = holes∧𝐼[n:end] * signflip
  subpseudo,sflip = swap_parity(vcat(sb,holes))
  # then adjust with sign of k⋅k
  sflip = sflip*swap_parity(vcat(sb,sb))[2]
  :(T($sflip)*sign(b)*abs(b*i))
end

⋆(b::B, i::Type{BI}) where {T, B<:Blade{T}, BI<:Blade} = ⋆(b, i(one(T)))

"""
    ⋆(k)

Hodge star operator mapping k to it's Hodge dual.
Defined by k∧⋆(k) == (k⋅k)*𝐼 where k is generated from orthonormal 1-vectors and 
𝐼 is the psuedoscalar for the generating vector space.
"""
⋆(b::B) where {T,B<:Blade{T}} = ⋆(b, pseudoscalar(b))

⋆(s::Number) = untype(⋆dual(1)*dual(1))(s)
⋆(s::Number, i::BI) where {BI<:Blade} = i
⋆(s::Number, i::Type{BI}) where {BI<:Blade} = ⋆(s, i(one(s)))

import LinearAlgebra: norm,norm_sqr,normalize

norm_sqr(b::K) where {K<:Blade} = magnitude(b)*magnitude(b)
norm(b::K) where {K<:Blade} = abs(magnitude(b))
normalize(b::K) where {T, K<:Blade{T}} = sign(magnitude(b))*one(T)∧untype(b)

"""
    basis_kblades(k, n)

a list of all basis blades of degree n for the algebra k belongs to.

i.e. basis_kblades(e₁, 2) will depend the dimension of the algebra e₁ was generated in
"""
@generated function basis_kblades(::Type{K}, d::Val{N}) where {N,K} 
  ns = parentmodule(K)
  n = grade((FunctionWrapper{Any,
                             Tuple{Type{K},}}(pseudoscalar))(K))
  kone = alle(ns,n)[1:n]
  b = filter(e->grade(e)==N, alle(ns, n))
  :($b)
end

basis_kblades(k::K, d::Val{N}) where {N,K} = basis_kblades(K, d)
basis_kblades(u,k::Int) = basis_kblades(u,Val(k))

"""
    basis_1blades(k)

a list of all basis 1-blades for the algebra k belongs to.

i.e. basis_1blades(e₁) will depend the dimension of the algebra e₁ was generated in
"""
basis_1blades( k::B ) where {B<:Blade} = basis_kblades(k,1)
basis_1blades( k::Type{B} ) where {B<:Blade} = basis_kblades(k,1)

"""
    factor(b)

all the smallest factors, that when wedged together create b.
"""
@generated function factor(b::B) where {B<:Blade}
  bs = subspace(b)

  :((b.x,basis_1blades(b)[$bs]...))
end

"Convert blade value to their type with parameter removed"
@generated function untype(b::B) where {T,B<:Blade{T}}
  ns = parentmodule(b)
  n = grade((FunctionWrapper{Any, 
                             Tuple{Type{b},}}(pseudoscalar))(b))
  es = alle(ns,n)
  es = vcat(es,raise.(es))
  une = es[findfirst(eᵢ->eᵢ{T} == b, es)]
  :($une)
end


"""
    outermorphism(L, b)

apply the linear transform L to the geometric algebra object b as an outermorphism.

the outermorphism property
L(a∧b∧c) = L(a)∧L(b)∧L(c)
"""
function outermorphism(L, b::B) where B<:Blade
  sb = factor(b) # (scalar, followed by all 1-blades)
  𝐼 = pseudoscalar(b)  # need this to construct KVector from coords
  mapreduce(bᵢ->KVector(L*coords(one(bᵢ)), 𝐼), ∧, sb[2:end])*sb[1]
end

Base.in(be::B, bs::B2) where {B<:Blade, B2<:Blade} = (subspace(B) ∩ subspace(B2)) == subspace(B)

