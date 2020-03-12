[![Build Status](https://travis-ci.com/mewertd2/Multivectors.jl.svg?branch=master)](https://travis-ci.com/mewertd2/Multivectors.jl)
[![codecov.io](https://codecov.io/github/mewertd2/Multivectors.jl/coverage.svg?branch=master)](https://codecov.io/github/mewertd2/Multivectors.jl?branch=master)

# Multivectors

The `Multivectors` [Julia](http://julialang.org) package defines the `Multivector` Type
to represent mixed-grade linear combinations of [KVectors](./KVectors_README.md), which are in turn a vector space of [Blades](./Blades_README.md) of a given grade.
`Multivectors` is intended to be an implementation of [Geometric Algebra](https://en.wikipedia.org/wiki/Geometric_algebra), although it is useful for any Clifford algebra.  
Where operator or naming conventions differ, the ones from Geometric Algebra most closely aligned to conventions used in Computer Science will be used.

It is recommended to read the documentation on [Blades](./Blades_README.md) and [KVectors](./KVectors_README.md) first.

## Geometric Product

`Multivectors` essentially extends the algebras and Types defined as [Blades](./Blades_README.md) and [KVectors](./KVectors_README.md) with `*`, the [Geometric Product](https://en.wikipedia.org/wiki/Geometric_algebra#The_geometric_product).

There are many other operators defined, but the geometric product is fundamental.  In fact, we could go back and redefine the wedge `∧` and inner products `⋅` using the geometric product.

    a⋅b = ½(a*b + b*a) 
    a∧b = ½(a*b - b*a)

Note: this is only strictly true for `a` and `b` as 1-vectors.

To complete the picture for `*` and extend to `Multivectors` we require a grade projection operator.  The grade projection operator simply returns the k-vector of grade k contained in the `Multivector`.

    ⟨A⟩ᵢ = grade(A,i)

and commutator product to catch any (most) grades not in `∧` and inner products `⋅`

    A×B = ½(A*B-B*A) 

With grade projection we can define wedge `∧` and inner products `⋅` using the geometric product of `Multivectors`.  Notice the symmetry where one is grade raising and the other grade lowering.

    A∧B = grade(A*B, grade(B)+grade(A)) # grade raising
    A⋅B = grade(A*B, grade(B)-grade(A)) # grade lowering

## Inner Product

The inner product `⋅` is defined in `Multivectors` to be the left contraction operator ⌋, `lcontraction`.
The contraction `a⋅B` between blades `a` and `B` will result in a blade with grade `grade(B)-grade(a)` that is orthogonal to `a` and contained in `B`.  
These 3 properties ( grade reduction, orthogonality and projection ) in one operator make left contraction powerful but less intiutive than the standard inner product you may be used to.
This is a generalization of the inner product ( dot product ) from vector algebra to blades and multivectors.

## Examples

### Barycentric coordinates

    julia> using Multivectors

    julia> @generate_basis("+++", true)  # generate blades for euclidean 3D-space

    julia> a = 0.0e₁+0.0e₂; b = 1.0e₁ + 0.0e₂; c = 0.0e₁ + 1.0e₂;  # a simple right angle triangle

    julia> A = (b-a)∧(c-a)  # twice the area of the triangle. we don't worry about the factor of 2

Make a function to calculate barycentric coords as the ratio of the area of a triangle made with a point `p` and an edge over original triangle.  i.e. the barycentric coord for vertex `a` is the ratio Δpbc/Δabc

    julia> barycoords(p) = ((c-b)∧(p-b)/A, (a-c)∧(p-c)/A, (b-a)∧(p-a)/A)  # a tuple of coords

Notice how the code very directly represents the geometric relationship.  The body of the function is also coordinate free ( we never index into the points or vertices ).

    julia> barycoords(0.0e₁)
    (1.0, -0.0e₁₂, -0.0e₁₂)

    julia> barycoords(1.0e₁)
    (-0.0e₁₂, 1.0, -0.0e₁₂)

    julia> barycoords(0.5e₁+0.5e₂)
    (0.0, 0.5, 0.5)

    julia> barycoords(0.1e₁+0.25e₂)
    (0.65, 0.1, 0.25)

    julia> barycoords(0.1e₁+0.25e₂+10.0e₃)  # go off-plane by adding +10 in "up" direction
    (Multivector{Float64,2}
    ⟨0.65⟩₀ + ⟨10.0e₁₃,10.0e₂₃⟩₂, Multivector{Float64,2}
    ⟨0.1⟩₀ + ⟨-10.0e₁₃,-0.0e₂₃⟩₂, Multivector{Float64,2}
    ⟨0.25⟩₀ + ⟨0.0e₁₃,-10.0e₂₃⟩₂)
        
When we go off-plane, we get a very natural result where the barycentric coords are in the grade 0 part of a multivector.  As a bonus there is extra information in the higher grade k-vectors.
That extra info in the higher grades is a bit odd (it's the Lie bracket of the ratio operator). 
We really just want the scalar part of the multivector.  
Let's clean up the results by selecting the grade 0 scalar explicitly.
Selecting the grade with the relevant results is a common pattern when working with Multivectors.

    julia> baryscalars(p) = map(k->grade(k, 0), barycoords(p))

    julia> baryscalars(0.1e₁+0.25e₂+10.0e₃)
    (0.65, 0.1, 0.25)

    julia> baryscalars(1.0e₁)
    (0.0, 1.0, 0.0)

Extending to higher dimensions is straightforwards.
Tetrahedron.

    julia> d = 1.0e₃

    julia> V = A∧d
    1.0e₁₂₃

    julia> barycoords4(p) = ((c-b)∧(p-b)∧(d-c)/V, 
                             (a-c)∧(p-c)∧(d-a)/V, 
                             (b-a)∧(p-a)∧(d-a)/V, 
                             (b-a)∧(p-a)∧(a-c)/V)
    barycoords4 (generic function with 1 method)

    julia> barycoords4(0.25e₁+0.25e₂+0.25e₃)
    (0.25, 0.25, 0.25, 0.25)

    julia> barycoords4(0.1e₁+0.2e₂+0.3e₃)
    (0.39999999999999997, 0.1, 0.2, 0.3)

Add an extra dimension to get barycentric coords of a 4D volume.   Now we are doing something you can't do directly with a cross product.  Cross product doesn't exist in 4D.

Note, we need to restart julia to clear out the old basis from the Main module before we generate a new basis.
Better practice is to namespace a basis within it's own module.

Behold, the barycentric coordinates of a pentachoron.

    julia> using Multivectors

    julia> module R4
             using Multivectors
             @generate_basis("++++")  # 4D euclidean space
           end

    julia> using .R4

    julia> e₁, e₂, e₃, e₄ = alle(R4,4)[1:4]  # pick out the basis blades we will work with

    julia> a = 0.0e₁+0.0e₂; b = 1.0e₁; c = 1.0e₂; d = 1.0e₃; e = 1.0e₄ # a pentachoron!

    julia> H = b∧c∧d∧e  # hypervolume ( a is at origin )

    julia> barycoords5(p) = ((c-b)∧(p-b)∧(d-c)∧(e-b)/H, 
                             (a-c)∧(p-c)∧(d-a)∧(e-a)/H, 
                             (b-a)∧(p-a)∧(d-a)∧(e-b)/H, 
                             (b-a)∧(p-a)∧(a-c)∧(e-b)/H,
                             (b-a)∧(p-a)∧(d-a)∧(a-c)/H)

    julia> barycoords5(1.0e₄)
    (0.0, 0.0e₁₂₃₄, 0.0e₁₂₃₄, 0.0e₁₂₃₄, 1.0)

    julia> barycoords5(1.0e₁+1.0e₄)
    (-1.0, 1.0, 0.0e₁₂₃₄, 0.0e₁₂₃₄, 1.0)

    julia> barycoords5(0.0e₄)
    (1.0, 0.0e₁₂₃₄, 0.0e₁₂₃₄, 0.0e₁₂₃₄, 0.0e₁₂₃₄)

    julia> barycoords5(0.1e₁+0.1e₄)
    (0.8, 0.1, 0.0e₁₂₃₄, 0.0e₁₂₃₄, 0.1)

    julia> barycoords5(0.1e₁+0.2e₂+0.2e₃+0.4e₄)
    (0.09999999999999992, 0.1, 0.2, 0.2, 0.4)

## Quaternions, Rotors, Versors

Quaternions have a particularly simple construction in Geometric Algebra.
`q = a/b`
Geometrically this is asking `q` to be a mulitivector that would transform `b` into `a` via the geometric product.
`q` will also have the effect of rotating any vector `c` laying in the plane `a∧b` by the amount needed to rotate `b` into `a`.
We assume `q` is normalized.
In order for the `q` to act on (multi)vectors not neccesarily in the plane of rotation, we treat it as a versor.

A quaternion is a versor, simply means that you use the sandwich product when transforming an object with it.
Using the sandwich product we multiply by `q` twice and will end up rotating by twice the angle.  So we modify our initial construction rule to `q = normalize(â+b̂)/b̂`.

Now we get a familiar quaternion transformation rule `q̃*v*q`.  
Where we use `q̃` to indicate the reverse of `q`, which acts like complex conjugation and flips the sign of grade 2 blades.  
This construction extends to higher and lower dimensions and doesn't involve complex numbers.  
Therefore it's called a rotor in geometric algebra and not a quaternion.
Geometrically it is better to view rotors as a sequence of reflections to understand how it operates on vectors not parallel to the plane of rotation.

### Example

Construct a quaternion/rotor taking 1-vector 1.0e₁ to vector half45

    # normalized vector half-way between x and x+y
    julia> half45 = normalize(1.0e₁ + normalize(1.0e₁+1.0e₂))

    julia> q = half45/1.0e₁

Transform a 1-vector with the sandwich product.

    julia> v = reverse(q)*(1.0e₁+1.0e₂+1.0e₃)*q

    julia> grade(v, 1) |> prune∘sortbasis
    2-element KVector{Float64,1,2}:
     1.414213562373095e₁
                   1.0e₃

Rotors can be constructed using half-angle of trig functions, like quaternions.

    julia> cos(π/8) - sin(π/8)*1.0e₁₂ == q
    true

## Other Operators

Depending on the specific Geometric Algebra in use it may be desireable to define other operators.  For example `meet` and `join` operators are very useful, but will differ depending on the context.  Where there are multiple possible definitions/implementations of an operator, `Multivectors` chooses to omit such an operator rather than include it.

Most operators and methods defined for [KVectors](./KVectors_README.md) and [Blades](./Blades_README.md) work on `Multivectors` through either linearity (extended via vector space `+` and scalar `*` ) or outermorphism (extended via `∧`).

## Performance and Design

Blades, KVectors and Multivectors, in it's current iteration, is designed for exploring and prototyping novel algorithms in the emerging field of Applied Geometric Algebra with a focus on Computer Base Animation ( CGI ).  While the foundational Types are optimally performant, some work is needed to extend that performance through the rest of the Types.

[Blades](./Blades_README.md) have been designed and implemented with optimal performance in mind.  Operations on simple Blades have around the same performance as similar operations on native scalars.

    julia> x = sqrt(2.0); y = exp(1.0); ex = e₂₃(x); ey = e₁₃(y);

    julia> @btime x*y
      23.008 ns (1 allocation: 16 bytes)
    3.844231028159117

    julia> @btime ex*ey
      26.635 ns (1 allocation: 16 bytes)
    3.844231028159117e₁₂

This is accomplished with Julia's metaprogramming features.  We effectively leverage Julia's Type system and compiler to do the heavy lifting for us by creating unqiue Types for each Blade.

    julia> (typeof(ex), typeof(ey), typeof(ex) == typeof(ey))
    (e₂₃{Float64}, e₁₃{Float64}, false)

This performance minded design has not been extended to KVectors or Multivectors.  
The intention is to keep Blades, KVectors and Multivectors as a general reference implementation of Geometric Algebra.  
The main hurdle to achieving performance is to give Julia enough Type information to effectively optimize the code while still maintaining flexibility and ease of use for our Types.
Since we rely on the Julia compiler/parser to achieve performance, it could be that future versions of Julia will optimize KVectors and Multivectors (to a certain extent) for us.

Future versions or new packages implementing KVectors and Multivectors will be performant.  

Truely great performance will likely require specializing on a fixed algebra or set of objects (for an example of this approach: [Klein](https://github.com/jeremyong/Klein)).

## Related Packages/Types

See the documentation of [KVectors](./KVectors_README.md) and [Blades](./Blades_README.md) for more information.

[Grassmann](https://github.com/chakravala/Grassmann.jl) is another julia package that implements a Geometric Algebra in the context of a wider algebraic framework.

## Project Information

### Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

### Authors

* **Michael Alexander Ewert** - Developer - [Digital Domain](https://digitaldomain.com)

### License

This project is licensed under a modified Apache 2.0 license - see the [LICENSE](./LICENSE) file for details
