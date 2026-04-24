+------------------------------------------------------------------------------+
| PYQUANT: Simple PYthon package for the simulation of QUANTum systems         |
|                                                                              |
| Author of this file: Simon Euchner                                           |
+------------------------------------------------------------------------------+


Description.

    While working on my master's thesis in the year 2025, it became clear to me
that for the basic simulations I run on a daily basis, I always need the same
numerical quantities. Therefore, I decided to write a small software library
to avoid rewriting the same code repeatedly.
    To implement this library, I chose the scripting language Python, since it
offers an easy-to-use interface to well-tested and fast numerical routines.
I took inspiration from packages like QuTiP [https://qutip.org], but decided to
create my own solution for several reasons:

    1. I find it difficult to understand how exactly routines work in larger
       frameworks like QuTiP.

    2. Large packages often include complex help mechanisms and abstraction
       layers that can complicate optimisation.

    3. I want to use sparse matrices at ALL times. I personally believe that,
       except for heavy numerics, there is rarely a reason to use dense
       matrices. Sparse routines are typically fast enough for the types of
       simulations this library is intended for.

    All functions are placed in a single file to make the package structure
as transparent as possible.


Dependencies.

 > scipy
 > numpy


Documentation.

    All documentation is built directly into the library. You can access it by
calling Python's 'help()' method on any object or function.


QuTiP compatibility.

    If you have a quantum object defined by PYQUANT, you can convert it to a
QuTiP object by passing its data to the QuTiP constructor:

    QuTiP_QOBJ = qutip.Qobj(PYQUANT_QOBJ.data)

Now you can use the QuTiP object as if PYQUANT never existed, ensuring
seamless collaboration with others using QuTiP.


Installation.

    To install the package locally, navigate to the project directory and use
pip.
    IMPORTANT: Since the name 'pyquant' is already taken on PyPI by an unrelated
project, you must ensure you are installing from the local directory by
including the '.' (dot):

    cd PYQUANT
    pip install .

    To modify the source code, it is recommended to use editable mode:

    pip install -e .
