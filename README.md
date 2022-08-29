# discBar
C++ and python code for potential-density models for galactic bars according to Dehnen & Aly (2022, MNRAS)

C++:
files discBar.h and discBar.cc.
Compile diskBar.cc to create discBar.o using make or any C++17 compiler. Link your programs against discBar.o
capability: potential, forces, force derivatives, density (no optimised)
useful for integrating orbits

python:
file discBar.py
simply import and go. Has Doc strings for almost everything.
capability: potential, forces, density, surface density
useful for fitting/comparing models and making plots
