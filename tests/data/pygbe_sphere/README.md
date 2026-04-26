# pyGBe sphere test fixtures

These mesh and charge files (`sphere500_R4.{vert,face}`,
`offcenter_R2.pqr`) are derived from [pyGBe][pygbe]'s sphere
regression-test data and bundled here so selfie's
`pygbe_sphere_mesh` integration test can run without a network
fetch.

[pygbe]: https://github.com/pygbe/pygbe

When citing pyGBe, please use the JOSS paper:
[Cooper et al., *J. Open Source Softw.* **1**(4), 43 (2016)](https://doi.org/10.21105/joss.00043).

## Upstream license

pyGBe is distributed under the BSD 3-Clause License. The license
requires that the original copyright notice, the list of conditions,
and the disclaimer be reproduced in any redistribution. Reproduced
here to satisfy that requirement:

> Copyright (c) 2013-2015 by Christopher Cooper, Lorena Barba
> Copyright (c) 2016 by Christopher Cooper, Natalia Clementi,
> Gilbert Forsyth, Lorena Barba
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions
> are met:
>
> 1. Redistributions of source code must retain the above copyright
>    notice, this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above
>    copyright notice, this list of conditions and the following
>    disclaimer in the documentation and/or other materials provided
>    with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its
>    contributors may be used to endorse or promote products derived
>    from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
> "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
> FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
> COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
> INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
> HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
> STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
> OF THE POSSIBILITY OF SUCH DAMAGE.

## Files

- `sphere500_R4.vert`, `sphere500_R4.face` — MSMS-format triangulation
  of a 500-face sphere of radius 4 Å. Source:
  pyGBe's `examples/sphere_dirichlet/` regression directory.
- `offcenter_R2.pqr` — single unit charge offset from the origin
  inside the sphere; matches pyGBe's exterior-charge convergence
  test setup.

## Selfie's contribution

Selfie does not modify these files; they are reproduced verbatim
from upstream.
