[![Build Status](https://travis-ci.org/gasagna/BorderedArrays.jl.svg?branch=master)](https://travis-ci.org/gasagna/BorderedArrays.jl)

# BorderedArrays.jl
This package provides a basic interface to solve special systems of linear equations where the lhs matrix is bordered, i.e. systems of the type

\begin{equation}
M x = r
\end{equation} 

where $M$ is made up as

\begin{equation}
 \begin{pmatrix}
  A   & b \\
  c^T & d \\
 \end{pmatrix}
\end{equation}
