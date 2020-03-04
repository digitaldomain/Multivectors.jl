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

using Test
using Multivectors
using KVectors
using Blades
using LinearAlgebra


module PG2
  using Blades, KVectors, Multivectors, LinearAlgebra
  @generate_basis("++0")

  const eₒ_index = 3
  const eₒ = e₃
  const e₂ₒ = e₂₃
  const e₁ₒ = e₁₃

  meet(a, b) = a∧b
  point(x::Real, y::Real) = x*e₂ₒ + y*e₁ₒ + one(x)*e₁₂
  direction(x::Real, y::Real) = x*e₂ₒ + y*e₁ₒ
  point( v::KVector{T} ) where T = point(coords(v)[1:2]...)
  line(a,b) = a∨b

  """ project point to euclidean (1eₒ) plane """ 
  projr(a) = a/(dual(1.0e₃)⋅a)

  function circumcentre( a, b, c )
    ab = line(a, b)
    bc = line(b, c)
    ab_mid = 0.5*(a+b)
    bc_mid = 0.5*(b+c)
    abperp = ab⋅ab_mid
    bcperp = bc⋅bc_mid

    projr(abperp∧bcperp) 
  end
  
end

using .PG2
@testset "PG2" begin
  @test PG2.circumcentre( PG2.point(0.0, 0.0), PG2.point(1.0,0.0), PG2.point(0.0,1.0) ) == PG2.point(0.5, 0.5)
end

module HomogeneousG2
  using Blades, KVectors, Multivectors, LinearAlgebra
  @generate_basis("+++")

  const eₒ_index = 3
  const eₒ = e₃

  ishomog( v::KVector{T} ) where T = filter(x->x==eₒ_index, subspace.(v)) |> isempty
  point( a, b ) = a*e₁+b*e₂ + one(a)*e₃
  point( v::KVector{T} ) where T = ishomog(v) ? v : v + one(T)eₒ
  line( p, q ) = p∧q
  plane( p, q, r ) = p∧q∧r
  unitize( p::KVector{T} ) where T = (α = one(T)eₒ⋅p; iszero(α) ? p : p*inv(alpha))  

  function meet(a, b)
    J = inv(1.0e₁₂₃)
    ((a*J)∧(b*J))*J
  end

  """ project to euclidean (1eₒ) plane """ 
  projr(a) = a/norm(1.0eₒ⋅a)

  function circumcentre( a, b, c )
    mid_ab = 0.5*(a+b)
    mid_bc = 0.5*(b+c)
    ab = (a∧b)
    bc = (b∧c)
    rab = 1.0eₒ⋅ab
    rbc = 1.0eₒ⋅bc
    tri = rab∧rbc
    abperp = rab*inv(tri)
    bcperp = rbc*inv(tri)
    centre_ab = mid_ab∧abperp
    centre_bc = mid_bc∧bcperp
    meet(centre_ab, centre_bc)
  end
end

using .HomogeneousG2
@testset "HomogeneousG2" begin
  pointh = HomogeneousG2.point
  @test HomogeneousG2.circumcentre( pointh(0.0, 0.0), pointh(1.0,0.0), pointh(0.0,1.0) ) == pointh(0.5, 0.5)
end

module CGA
  using Blades, KVectors, LinearAlgebra
  
  @generate_basis("++++-",false,true)
  const eo = 0.5*(1.0e₄ + 1.0e₅)
  const e∞ = 1.0e₅-1.0e₄

  point( p::K ) where {T, K<:Union{KVector{T,1}, Blade{T,1}}} = one(T)*eo + p + (one(T)/2)*p*p*e∞
  locate( a, M ) = (a⋅M)/M
  circumcentre( a, b, c) = locate(a, (a∧b + b∧c + c∧a)/(a∧b∧c∧e∞))
end

using .CGA

@testset "CGA3D" begin
  e₁, e₂, e₃ = alle( CGA, 5 )[1:3]
  using .CGA: eo, e∞, point, circumcentre
  @test e∞⋅e∞ == 0
  @test eo⋅e∞ == -1.0

  # test circumcentre of triangle
  
  a = point(0.0e₁+0.0e₂)
  b = point(1.0e₁)
  c = point(1.0e₂)

  C = (a∧b + b∧c + c∧a)/(a∧b∧c∧e∞)
  ccentre = (a⋅C)/C
  @test coords(grade(ccentre,1))[1:2] == [0.5,0.5]
  a = point(0.0e₁+0.0e₂)
  b = point(1.0e₁)
  c = point((sqrt(3.0)/2.0)e₂)
  ccentre = grade(circumcentre(a,b,c), 1) 

  @test norm(ccentre - grade(a,1)) ≈ norm(ccentre - grade(b,1)) ≈ norm(ccentre - grade(c,1))
end

module PG3
  using Blades
  @generate_basis("+++0")
end
using .PG3
@testset "PGA3D" begin
  e₁, e₂, e₃, e₄ = alle( PG3, 4)[1:4]
  e₁₂ = PG3.e₁₂; e₃₄ = PG3.e₃₄; e₁₂₃₄ = PG3.e₁₂₃₄;
  a = e₁(1.0); b = e₂(2.0); c = e₃(3.0); d = e₄(4.0)
  @test typeof(a+b*c) == typeof(Multivector{Float64,2}())
  B1 = KVector(a)
  B2 = a*b + b*c
  B3 = a*b*c + b*c*d
  B4 = KVector(a*b*c*d)
  M13 = B1+B3
  @test grade(M13) == 3
  @test grades(M13) == [1,3]
  @test grade(M13, 2) == 0.0
  @test grade(M13, 1) == B1
  @test grade(M13, 3) == B3
  @test grade(M13, 0) == 0.0
  M013 = M13+42.0
  @test grade(M013, 0) == 42.0
  B = a+b
  @test B*B == a*a+a*b+b*a+b*b
  @test grade((2e₂+3e₃)*(2e₂+3e₃), 0) == 2*2+3*3

  A = 2e₃ +3e₂
  B = 5e₄+6e₃
  # if we had an e₁ (degen metric) component this would not be true.
  @test A⋅!B == !(A∧B)
  
  A₀ = 2e₁ + 3e₂
  B₀ = 5e₁ + 6e₃

  # interesting though, this is ok.  
  𝐼 = 1e₁₂₃₄ 
  @test A₀⋅(B₀⋅𝐼) == (A₀∧B₀)⋅𝐼 == (A₀∧B₀)*𝐼 == A₀⋅(B₀*𝐼)# == A₀⋅(B₀*dual(1))

  C = 1e₁₂+4e₃₄
  @test (A∧B)⋅C == A⋅(B⋅C)

  # inner product tests
  # conjugate symmetry? a property an innerproduct is supposed to have.  probably not
  #  @test A⋅C == conj(C⋅A)
  # linearity in first argument 
  @test (A+B)⋅C == (A⋅C)+(B⋅C) 

  @test 2*A×B == (A*B-B*A)
  A = 2-A
  B = 2*B + 3(e₂∧e₃)
  @test 2A×B == (A*B-B*A)
end

module G4
  using Blades
  @generate_basis("++++")
end
using .G4
@testset "contraction" begin

  e₁, e₂, e₃, e₄  = alle( G4, 4)[1:4]

  rand1() = rand(1)[1]
  a = rand1()∧e₁ + rand()∧e₃
  # test lcontraction result is orthogonal to original
  B = rand()*(e₁∧e₂)
  @test a⋅(a⋅B) == 0.0
  B = B + rand()*(e₃∧e₄)
  @test a⋅(a⋅B) == 0.0
  

  A = 1.0 + 3.0G4.e₁₃
  @test lcontraction(a,A) == a⋅A
  @test rcontraction(A,a) == (A*a)[grade(A)-grade(a)]
  @test grades(A) == [0,2]
  show(A)
  @test first(A) == 1.0
  @test [i for i in A] == map(i->i, A) |>collect == (i for i in A) |> collect 
  @test length(A) == 2
  @test mapreduce(identity, +, A) == A
  @test mapreduce(conj, +, A) == conj(A)
  @test 1.0e₁ + A[2] + A[0] == A + 1.0e₁ 
  @test iszero(grade(A,10))
  @test grades(1.0) == [0]
  @test grade(1.0, 0) == 1.0
  @test grade(1.0, 1) == 0.0
  @test grades(A[2]) == [2]
  @test prune(dual(A)*pseudoscalar(A[2])) == A
  @test A/2.0 == A*0.5 
  @test A[2] == A-A[0]
  @test mapreduce(LinearAlgebra.norm_sqr, +, A) == scalarprod(A,reverse(A)) == 10.0
  @test normalize(A) == A/sqrt(scalarprod(A,reverse(A)))

  for i in A
    show(i)
  end

end

module G3
  using Blades
  @generate_basis("+++",false,true,true)
end
using .G3
@testset "Quaternion" begin

  e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃ = alle( G3, 3)
#==
  Quaternion multiplication
×	i	j	k
i	−1	k	−j
j	−k	−1	i
k	j	−i	−1
==#

  # quaternion basis
  𝑖 = e₂₃; 𝑗 = e₁₃; 𝑘 = e₁₂
  @test 1𝑖*1𝑗*1𝑘 == -1

  @test (1𝑖*1𝑖, 1𝑖*1𝑗, 1𝑖*1𝑘) == (-1, 1𝑘, -1𝑗)
  @test (1𝑗*1𝑖, 1𝑗*1𝑗, 1𝑗*1𝑘) == (-1𝑘, -1, 1𝑖)
  @test (1𝑘*1𝑖, 1𝑘*1𝑗, 1𝑘*1𝑘) == (1𝑗, -1𝑖, -1)

  𝑖 = e₁₂; 𝑗 = e₂₃; 𝑘 = e₁₃
  @test 1𝑖*1𝑗*1𝑘 == -1
  @test (1𝑖*1𝑖, 1𝑖*1𝑗, 1𝑖*1𝑘) == (-1, 1𝑘, -1𝑗)
  @test (1𝑗*1𝑖, 1𝑗*1𝑗, 1𝑗*1𝑘) == (-1𝑘, -1, 1𝑖)
  @test (1𝑘*1𝑖, 1𝑘*1𝑗, 1𝑘*1𝑘) == (1𝑗, -1𝑖, -1)


  half45 = normalize(1.0e₁ + normalize(1.0e₁+1.0e₂))
  q = half45/1.0e₁

  # Transform a 1-vector with the sandwich product.
  v = reverse(q)*(1.0e₁+1.0e₂+1.0e₃)*q

  v′ = grade(v, 1) |> prune∘sortbasis
  @test v′⋅1.0e₃ == 1.0
  @test v′⋅1.0e₁ ≈ sqrt(2.0)

  # Rotors can be constructed using half-angle of trig functions, like quaternions.

  @test cos(π/8) - sin(π/8)*1.0e₁₂ == q

end

@testset "Barycentric" begin

  e₁, e₂, e₃  = alle( G3, 3)[1:3]

  a = 0.0e₁+0.0e₂; b = 1.0e₁ + 0.0e₂; c = 0.0e₁ + 1.0e₂;  # a simple right angle triangle

  A = (b-a)∧(c-a)  # twice the area of the triangle. we don't worry about the factor of 2

  # Make a function to calculate barycentric coords as the ratio of the area of a triangle made with a point `p` and an edge over original triangle.  i.e. the barycentric coord for vertex `a` is the ratio Δpbc/Δabc

  barycoords(p) = ((c-b)∧(p-b)/A, (a-c)∧(p-c)/A, (b-a)∧(p-a)/A)  # a tuple of coords

  # Notice how the code very directly represents the geometric relationship.  The body of the function is also coordinate free ( we never index into the points or vertices ).

  @test barycoords(0.0e₁)[1] == 1.0
  @test barycoords(1.0e₁)[2] == 1.0
  @test barycoords(0.5e₁+0.5e₂) == (0.0, 0.5, 0.5)

  baryscalars(p) = map(k->grade(k, 0), barycoords(p))

  @test baryscalars(0.1e₁+0.25e₂+10.0e₃) == (0.65, 0.1, 0.25)
  @test baryscalars(1.0e₁) == (0.0, 1.0, 0.0)

  d = 1.0e₃
  V = A∧d

  # tetrahedron
  barycoords4(p) = ((c-b)∧(p-b)∧(d-c)/V, 
                    (a-c)∧(p-c)∧(d-a)/V, 
                    (b-a)∧(p-a)∧(d-a)/V, 
                    (b-a)∧(p-a)∧(a-c)/V)

  @test barycoords4(0.25e₁+0.25e₂+0.25e₃) == (0.25, 0.25, 0.25, 0.25)
end

