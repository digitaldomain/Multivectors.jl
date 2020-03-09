# KVectors

The `KVector` Type represents linear combinations of [k-blades](https://en.wikipedia.org/wiki/Blade_(geometry)).

All Blades in a given KVector have the same grade.

This is the main object in [Grassmann Algebra](https://en.wikipedia.org/wiki/Exterior_algebra).

KVectors essentially extends the algebras and Types defined in [Blades](./Blades_README.md) with the `+` operator.  This allows for a vector space of Blades.  

Most operators that act on Blades can act on KVectors.  Notable exceptions are the inner product `⋅` and geometric product `*`.
These can not, in general, operate on KVectors as they could result in a mixed grade vector.  The KVectors algebra is not closed under such operators.  For mixed grades you need [Multivectors](./REAME.md).

See the documentation of [Blades](./Blades_README.md) for more information.
